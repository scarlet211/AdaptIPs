import torch
import torch.nn as nn
import torch.optim as optim
from rdkit.VLib.NodeLib.demo import output
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification, RobertaForSequenceClassification
import numpy as np
import random
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold
import time
import pickle
import torch.nn.functional as F
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from model1 import Adapt_emb_CNNLSTM_ATT
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_token2index(sequences):
    token2index = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    print(token2index)

    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:

        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    print('sequences_residue', sequences[0:5])
    print('token_list', token_list[0:5])
    return token_list, max_len
def make_data_with_unified_length(token_list, max_len):
    token2index = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]  
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])

    print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
    print('max_len + 2', max_len)
    print('token_list + [pad]', token_list[0:5])

    return data
def esmcmain(client: ESMCInferenceClient, seq):
    # ================================================================
    # Example usage: one single protein
    # ================================================================
    protein = ESMProtein(seq)  # Initialize the ESMC protein sequence object

    # Use logits endpoint. Using bf16 for inference optimization
    protein_tensor = client.encode(protein)  # Convert the sequence into an index
    output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    assert isinstance(
        output, LogitsOutput
    ), f"LogitsOutput was expected but got {output}"
    assert output.logits is not None and output.logits.sequence is not None
    assert output.embeddings is not None and output.embeddings is not None
    print(
        f"Client returned logits with shape: {output.logits.sequence.shape} and embeddings with shape: {output.embeddings.shape}"
    )
    return output.embeddings
def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_acc = 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        y_pred = (np.array(y_prob) >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc
def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)          # python hash
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                  # multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
def get_scheduler(optimizer, name="cosine", **kwargs):
    name = name.lower()
    if name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1))
    if name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=kwargs.get("mode","min"),
                                                    factor=kwargs.get("factor",0.5), patience=kwargs.get("patience",3),
                                                    min_lr=kwargs.get("min_lr",1e-7))
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max",50), eta_min=kwargs.get("eta_min",0.0))
    if name == "cosine_restart":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=kwargs.get("T_0",10), T_mult=kwargs.get("T_mult",2))
    if name == "onecycle":
        # requires max_lr, steps_per_epoch, epochs
        return optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=kwargs["max_lr"],
                                            steps_per_epoch=kwargs["steps_per_epoch"],
                                            epochs=kwargs["epochs"],
                                            pct_start=kwargs.get("pct_start",0.3),
                                            anneal_strategy=kwargs.get("anneal_strategy","cos"),
                                            div_factor=kwargs.get("div_factor",25.0),
                                            final_div_factor=kwargs.get("final_div_factor",1e4))
    if name == "lambda_warmup_cosine":
        # warmup -> cosine decay (example)
        total_steps = kwargs["total_steps"]
        warmup_steps = kwargs.get("warmup_steps", int(0.1 * total_steps))
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")


def get_entropy(probs):
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent
def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent

def get_val_loss(logits, label, criterion):
    # logits: (b,1) or (b,C)
    alpha = 0.1
    
    try:
        ce = criterion(logits, label.float().view(-1))  # shape: (b,)
        probs_pos = torch.sigmoid(logits.view(-1, 1))            # (b,1)
        probs = torch.cat([1 - probs_pos, probs_pos], dim=1)     # (b,2) 
    except Exception:
      
        logits_cat = torch.cat([-logits, logits], dim=1)        # (b,2)
        ce = criterion(logits_cat.view(-1, 2), label.view(-1)) # shape: (b,)
        probs = F.softmax(logits_cat, dim=1)                   # (b,2)


    if ce.dim() == 0:
        ce = ce.unsqueeze(0)
    ce = ce.float().view(-1)

    transformed = (ce - alpha).abs() + alpha          # (b,)
    ent = get_entropy(probs).view(-1)                 # (b,)
    cond_ent = get_cond_entropy(probs).view(-1)       # (b,)

    sum_loss = transformed + ent - cond_ent           # (b,)
    return transformed
def train_and_evaluate_model_with_cv(train_data, train_label,test_data, test_label):

    # Device setting
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set random seed
    seed_val =42 #114514#3407#42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_seed(42)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../ESM2-150M")

    # Define metric parameters
    accuracies = []
    val_sensitivity_list = []
    val_specificity_list = []
    val_mcc_list = []
    val_auc_list = []

    # Data preprocessing
    encoded_texts = tokenizer(train_data, padding=True, truncation=True, return_tensors='pt')
    token_list, max_len = transform_token2index(train_data)
    fea = make_data_with_unified_length(token_list, max_len)
    fea1=torch.tensor(fea).to(device)

    # esmcmodel = ESMC.from_pretrained("esmc_600m")
    # embedding = []
    # count=0
    # res = [''.join(sub) for sub in train_data]
    # for i in res:
    #     te = esmcmain(esmcmodel, i).squeeze().cpu().detach().numpy()
    #     embedding.append(te)
    #     count += 1
    #     print(count)
    # fea2 = np.array(embedding)
    # np.save("fea2.npy", fea2)
    fea2 = np.load("fea2.npy")
    print(fea2.shape)
    fea2 = torch.tensor(fea2).to(device)

    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(train_label, dtype=torch.float32).unsqueeze(1).to(device)






    # =======================
    # Complete model training
    # =======================
    set_seed(42)
    num_epochs_final = 100
    batch_size = 32

    # all dataset
    full_dataset = TensorDataset(input_ids, attention_mask, labels, fea1, fea2)
    num_samples = len(full_dataset)

    indices = np.arange(num_samples)
    labels_np = labels.view(-1).cpu().numpy()  # (N,)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=42,
        stratify=labels_np
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset   = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)



    total_steps = len(train_loader) * num_epochs_final

    model = Adapt_emb_CNNLSTM_ATT().to(device)
    # criterion=nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    pos = labels.sum().item()
    neg = labels.shape[0] - pos
    pos_weight = torch.tensor(neg / (pos + 1e-12))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # criterion = nn.BCEWithLogitsLoss(pos_weight=2)
    # pos_weight=1.3
    # pos_weight_tensor = torch.tensor([pos_weight], device=device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


    lambda_contrast =0

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    # scheduler = get_scheduler(optimizer, name="cosine", T_max=40)
    # scheduler = get_scheduler(optimizer, name="reducelronplateau", factor=0.5, patience=3)


    # early stopping 
    best_auc_full = 0.0
    best_state_full = None
    best_acc_full = 0.0
    best_mcc_full = 0.0
    best_epoch_full = -1
    patience = 10
    patience_counter = 0

    print("===== Start full-data training =====")
    for epoch in range(num_epochs_final):
        # ---------- training ----------
        model.train()
        total_loss = 0.0

        for batch_input_ids, batch_attention_mask, batch_labels, batch_fea1, batch_fea2 in train_loader:
            optimizer.zero_grad()

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_fea1 = batch_fea1.to(device)
            batch_fea2 = batch_fea2.to(device)
            batch_labels = batch_labels.float().to(device).view(-1)  # (B,)

            logits, representation, fused, loss_contrast = model(
                batch_fea1, batch_fea2, batch_input_ids, batch_attention_mask, labels=batch_labels
            )
            # loss_main = criterion(logits, batch_labels)
            # loss = loss_main + lambda_contrast * loss_contrast

            loss = get_val_loss(logits, batch_labels, criterion)
            # loss=criterion(logits, batch_labels)
            # probs = torch.sigmoid(logits)
            # batch_probs = probs.cpu().numpy().reshape(-1)
            # batch_preds = (batch_probs >= 0.5).astype(int)


            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"[Full] Epoch {epoch + 1} | train_loss={avg_loss:.4f}")

        # ---------- early stopping ----------
        model.eval()
        val_predictions = []
        val_probabilities = []
        val_labels_epoch = []

        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_labels, batch_fea1, batch_fea2 in val_loader:
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_fea1 = batch_fea1.to(device)
                batch_fea2 = batch_fea2.to(device)
                batch_labels = batch_labels.float().to(device).view(-1)

                logits, representation, fused, loss_contrast = model(
                    batch_fea1, batch_fea2, batch_input_ids, batch_attention_mask, labels=None
                )
                probs = torch.sigmoid(logits)

                batch_probs = probs.cpu().numpy().reshape(-1)

                batch_preds = (batch_probs >= 0.5).astype(int)

                val_predictions.extend(batch_preds.tolist())
                val_probabilities.extend(batch_probs.tolist())
                val_labels_epoch.extend(batch_labels.cpu().numpy().tolist())



        val_mcc = matthews_corrcoef(val_labels_epoch, val_predictions)
        val_auc = roc_auc_score(val_labels_epoch, val_probabilities)

        TP = TN = FP = FN = 0
        for y_true, y_pred in zip(val_labels_epoch, val_predictions):
            if y_pred == 1 and y_true == 1:
                TP += 1
            elif y_pred == 0 and y_true == 0:
                TN += 1
            elif y_pred == 1 and y_true == 0:
                FP += 1
            else:
                FN += 1

        val_accuracy_full = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        sn = TP / (TP + FN + 1e-8)
        sp = TN / (TN + FP + 1e-8)

        print(
            f"[Full] Epoch {epoch + 1} | "
            f"val_acc={val_accuracy_full:.4f} | "
            f"val_auc={val_auc:.4f} | val_mcc={val_mcc:.4f}| sn={sn:.4f}| sp={sp:.4f}"
        )

        # early stopping：
        if val_mcc > best_mcc_full + 1e-4:
            best_auc_full = val_auc
            best_acc_full = val_accuracy_full
            best_mcc_full = val_mcc
            best_epoch_full = epoch + 1
            best_state_full = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Full] Early stopping at epoch {epoch + 1}")
                break


    # After training, restore to the parameters corresponding to the optimal AUC.
    if best_state_full is not None:
        model.load_state_dict(best_state_full)
        print(
            f"[Full] Best model at epoch {best_epoch_full}: "
            f"AUC={best_auc_full:.4f}, ACC={best_acc_full:.4f}, MCC={best_mcc_full:.4f}"
        )
    else:
        print("[Full] No best model found.")

    # save
    model_save = model.state_dict()
    torch.save(model_save, '../Result/ST_MMM_model_TEST.pth')
    print("Saved final full-data model to ../Result/ST_MMM_model_TEST.pth")




