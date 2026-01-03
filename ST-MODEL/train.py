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
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]  # 前
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
    protein = ESMProtein(seq)  # 初始化ESMC蛋白序列对象

    # Use logits endpoint. Using bf16 for inference optimization
    protein_tensor = client.encode(protein)  # 将序列转化为索引
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
    os.environ['PYTHONHASHSEED'] = str(seed)          # 固定 python hash
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                  # multi-GPU
    # PyTorch Deterministic 设置（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 更严格（PyTorch >=1.8），遇到非确定性操作会报错
    # torch.use_deterministic_algorithms(True)
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
    # logits: (b,1) 或 (b,C)
    alpha = 0.1
        # 尝试用 BCEWithLogitsLoss（输入形状可为 (b,) 或 (b,1)）
    try:
        ce = criterion(logits, label.float().view(-1))  # shape: (b,)
        probs_pos = torch.sigmoid(logits.view(-1, 1))            # (b,1)
        probs = torch.cat([1 - probs_pos, probs_pos], dim=1)     # (b,2) 供熵函数使用
    except Exception:
        # 若 criterion 为 CrossEntropyLoss(2 classes)，将单 logit 扩展为两类 logits
        logits_cat = torch.cat([-logits, logits], dim=1)        # (b,2)
        ce = criterion(logits_cat.view(-1, 2), label.view(-1)) # shape: (b,)
        probs = F.softmax(logits_cat, dim=1)                   # (b,2)


    # 确保 ce 为一维 per-sample
    if ce.dim() == 0:
        ce = ce.unsqueeze(0)
    ce = ce.float().view(-1)

    # per-sample 变换和熵项（假设 get_entropy/get_cond_entropy 返回 per-sample 向量）
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

    # labels_np = labels.view(-1).cpu().numpy().astype(int)
    # # Define K-fold cross-validation
    # k = 5
    # # kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # counts = np.bincount(labels_np)
    # if counts.min() < k:
    #     raise ValueError(f"Some class has fewer than {k} samples (counts={counts}). Reduce n_splits or rebalance data.")
    #
    # # Start cross-validation
    # # for fold, (train_indices, val_indices) in enumerate(kf.split(input_ids)):
    # for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(labels_np)), labels_np)):
    #     print("##############")
    #     print(f"Fold {fold + 1}:")
    #     print("##############")
    #     # Split training and validation sets
    #     train_input_ids, train_attention_mask, train_labels, train_fea1, train_fea2 = input_ids[train_indices], \
    #     attention_mask[train_indices], labels[train_indices], fea1[train_indices], fea2[train_indices]
    #     val_input_ids, val_attention_mask, val_labels, val_fea1, val_fea2 = input_ids[val_indices], attention_mask[
    #         val_indices], labels[val_indices], fea1[val_indices], fea2[val_indices]
    #
    #     batch_size = 32
    #
    #     # Create DataLoader
    #     train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels, train_fea1, train_fea2)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels, val_fea1, val_fea2)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    #     # Initialize variables and model
    #     val_accuracy = 0
    #     val_sensitivity = 0
    #     val_specificity = 0
    #
    #     best_model = None
    #     num_epochs = 30
    #     total_steps = len(train_loader) * num_epochs
    #     print("train_loader length:", len(train_loader))
    #     # Model parameter initialization
    #     # model = EsmForSequenceClassification.from_pretrained("../ESM2-150M", num_labels=1).to(device)
    #     model = Adapt_emb_CNNLSTM_ATT().to(device)
    #     # criterion = nn.BCELoss(size_average=False)
    #     criterion = nn.BCEWithLogitsLoss()
    #     lambda_contrast = 0
    #     # Define loss function and optimizer
    #     optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #     # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    #     num_warmup_steps = int(0.1 * total_steps)  # 10% 作为 warmup
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=total_steps
    #     )
    #
    #     best_val_auc = 0
    #     best_metrics = None
    #     # Train model
    #     for epoch in range(num_epochs):
    #         model.train()
    #         total_loss = 0
    #         for batch_input_ids, batch_attention_mask, batch_labels, batch_fea1, batch_fea2 in train_loader:
    #             optimizer.zero_grad()
    #             # outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #             # loss = outputs.loss
    #             # logits = outputs.logits
    #
    #             # fx, presention, _, loss2 = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)  # torch.Size([256, 1])
    #             # loss = criterion(fx, batch_labels.type(torch.FloatTensor).to(device))  # B
    #             # loss = loss
    #             batch_labels = batch_labels.float().to(device)  # 先放到 device
    #             batch_labels = batch_labels.view(-1)
    #             logits, representation, fused, loss_contrast = model(batch_fea1, batch_fea2, batch_input_ids,
    #                                                                  batch_attention_mask, labels=batch_labels)
    #             loss_main = criterion(logits, batch_labels)
    #             loss = loss_main + lambda_contrast * loss_contrast
    #
    #             total_loss += loss.item()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #
    #         avg_loss = total_loss / len(train_loader)
    #         print("  Average training loss: {0:.4f}".format(avg_loss))
    #
    #         # Validate model for each epoch
    #         val_predictions = []
    #         val_probabilities = []
    #         val_labels = []
    #         model.eval()
    #         # Tracking variables
    #         for batch_input_ids, batch_attention_mask, batch_labels, batch_fea1, batch_fea2 in val_loader:
    #             with torch.no_grad():
    #                 # outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #                 # fx, presention, _, loss2 = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)
    #                 logits, representation, fused, loss_contrast = model(batch_fea1, batch_fea2, batch_input_ids,
    #                                                                      batch_attention_mask, labels=None)
    #                 probs = torch.sigmoid(logits)
    #
    #                 # outputs = outputs.logits
    #             # outputs=fx
    #             batch_probs = probs.cpu().numpy()
    #             batch_preds = (batch_probs >= 0.5).astype(int)
    #
    #             val_predictions.extend(batch_preds.tolist())
    #             val_probabilities.extend(batch_probs.tolist())
    #             val_labels.extend(batch_labels.detach().cpu().numpy())
    #
    #         # Calculate MCC, AUC
    #         val_mcc = matthews_corrcoef(val_labels, val_predictions)
    #         val_auc = roc_auc_score(val_labels, val_probabilities)
    #         # Calculate sensitivity and specificity
    #         TP = TN = FP = FN = 0
    #         for i in range(len(val_labels)):
    #             if val_predictions[i] == 1 and val_labels[i] == 1:
    #                 TP += 1
    #             elif val_predictions[i] == 0 and val_labels[i] == 0:
    #                 TN += 1
    #             elif val_predictions[i] == 1 and val_labels[i] == 0:
    #                 FP += 1
    #             else:
    #                 FN += 1
    #         val_sensitivity = TP / (TP + FN + 1e-8)
    #         val_specificity = TN / (TN + FP + 1e-8)
    #         val_accuracy = (TP + TN) / (TP + TN + FP + FN)
    #         print(
    #             f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}")
    #         output_string = f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}\n"
    #
    #         if val_auc > best_val_auc:
    #             best_val_auc = val_auc
    #             best_metrics = (val_accuracy, val_sensitivity, val_specificity, val_mcc, val_auc)
    #
    #         with open("../Result/ST_output.txt", "a") as file:
    #             file.write(output_string)
    #     accuracies.append(best_metrics[0])
    #     val_sensitivity_list.append(best_metrics[1])
    #     val_specificity_list.append(best_metrics[2])
    #     val_mcc_list.append(best_metrics[3])
    #     val_auc_list.append(best_metrics[4])
    #     # 保存最终模型
    #     model_save = model.state_dict()
    #     torch.save(model_save, f'../Result/ST_MMM_model_{fold + 1}.pth')
    # print("Cross Validation Results:")
    # print(
    #     f"Average Accuracy: {np.mean(accuracies):.4f} | Average Sensitivity: {np.mean(val_sensitivity_list):.4f} | Average Specificity {np.mean(val_specificity_list):.4f} | Average MCC: {np.mean(val_mcc_list):.4f} | Average AUC: {np.mean(val_auc_list):.4f}")


    #
    # # Complete model training
    # # 全量
    # num_epochs = 30
    # batch_size = 32
    # lambda_contrast=0
    # train_dataset = TensorDataset(input_ids, attention_mask, labels,fea1,fea2)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # total_steps = len(train_loader) * num_epochs
    # # model = EsmForSequenceClassification.from_pretrained("../ESM2-150M", num_labels=1).to(device)
    # model = Adapt_emb_CNNLSTM_ATT().to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # num_warmup_steps = int(0.1 * total_steps)  # 10% 作为 warmup
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=total_steps
    # )
    #
    # model.train()
    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     for batch_input_ids, batch_attention_mask, batch_labels,batch_fea1 ,batch_fea2 in train_loader:
    #         optimizer.zero_grad()
    #         # outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #         # loss = outputs.loss
    #         # logits = outputs.logits
    #         batch_labels = batch_labels.float().to(device)  # 先放到 device
    #         batch_labels = batch_labels.view(-1)
    #         logits, representation, fused, loss_contrast = model(batch_fea1, batch_fea2, batch_input_ids,
    #                                                              batch_attention_mask, labels=batch_labels)
    #         # loss_main = criterion(logits, batch_labels)
    #         # loss = loss_main + lambda_contrast * loss_contrast
    #         loss = get_val_loss(logits, batch_labels, criterion)
    #
    #         # fx, presention, _, loss2 = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)  # torch.Size([256, 1])
    #         # loss = criterion(fx, batch_labels.type(torch.FloatTensor).to(device))  # B
    #         # loss = loss
    #
    #         total_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #
    #     avg_loss = total_loss / len(train_loader)
    #     print("  Average training loss: {0:.4f}".format(avg_loss))
    #
    # model_save = model.state_dict()
    # torch.save(model_save, '../Result/ST_MMM_model.pth')



    # # Define K-fold cross-validation   加入早停
    # labels_np = labels.view(-1).cpu().numpy().astype(int)
    # k = 5
    # # kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    # counts = np.bincount(labels_np)
    # if counts.min() < k:
    #     raise ValueError(f"Some class has fewer than {k} samples (counts={counts}). Reduce n_splits or rebalance data.")
    #
    # # Start cross-validation
    # # for fold, (train_indices, val_indices) in enumerate(kf.split(input_ids)):
    # for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(labels_np)), labels_np)):
    #     print("##############")
    #     print(f"Fold {fold + 1}:")
    #     print("##############")
    #     # Split training and validation sets
    #     train_input_ids, train_attention_mask, train_labels,train_fea1, train_fea2= input_ids[train_indices], attention_mask[train_indices], labels[train_indices], fea1[train_indices], fea2[train_indices]
    #     val_input_ids, val_attention_mask, val_labels,val_fea1,val_fea2 = input_ids[val_indices], attention_mask[val_indices], labels[val_indices], fea1[val_indices], fea2[val_indices]
    #
    #     batch_size = 32
    #
    # #     # Create DataLoader
    #     train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels,train_fea1,train_fea2)
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels,val_fea1,val_fea2)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    #     # Initialize variables and model
    #     val_accuracy = 0
    #     val_sensitivity = 0
    #     val_specificity = 0
    #
    #
    #
    #     best_model = None
    #     num_epochs = 100
    #     total_steps = len(train_loader) * num_epochs
    #     print("train_loader length:", len(train_loader))
    #     # Model parameter initialization
    #     # model = EsmForSequenceClassification.from_pretrained("../ESM2-150M", num_labels=1).to(device)
    #     model = Adapt_emb_CNNLSTM_ATT().to(device)
    #     # criterion = nn.BCELoss(size_average=False)
    #     # criterion = nn.BCEWithLogitsLoss()
    #     pos = labels.sum().item()
    #     neg = labels.shape[0] - pos
    #     pos_weight = torch.tensor(neg / (pos + 1e-12))
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    #     lambda_contrast = 0.5
    #     # Define loss function and optimizer
    #     # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
    #     # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    #     # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    #     num_warmup_steps = int(0.1 * total_steps)  # 10% 作为 warmup
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=total_steps
    #     )
    #
    #     best_acc = 0.0
    #     best_metrics  = None
    #     best_state = None
    #     patience = 10
    #     patience_counter = 0
    #     # Train model
    #     for epoch in range(num_epochs):
    #         model.train()
    #         total_loss = 0
    #         for batch_input_ids, batch_attention_mask, batch_labels,batch_fea1 ,batch_fea2 in train_loader:
    #             optimizer.zero_grad()
    #             # outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #             # loss = outputs.loss
    #             # logits = outputs.logits
    #
    #             # fx, presention, _, loss2 = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)  # torch.Size([256, 1])
    #             # loss = criterion(fx, batch_labels.type(torch.FloatTensor).to(device))  # B
    #             # loss = loss
    #             batch_labels = batch_labels.float().to(device)  # 先放到 device
    #             batch_labels = batch_labels.view(-1)
    #             logits, representation, fused, loss_contrast= model(batch_fea1, batch_fea2,batch_input_ids, batch_attention_mask,labels=batch_labels)
    #             # loss_main = criterion(logits, batch_labels)
    #             # loss = loss_main + lambda_contrast * loss_contrast
    #             loss = get_val_loss(logits, batch_labels, criterion)
    #             total_loss += loss.item()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #
    #         avg_loss = total_loss / len(train_loader)
    #         # print("  Average training loss: {0:.4f}".format(avg_loss))
    #         print(f"Epoch {epoch + 1} | Average training loss: {avg_loss:.4f}")
    #
    #         # Validate model for each epoch
    #         val_predictions = []
    #         val_probabilities = []
    #         val_labels_epoch = []
    #         model.eval()
    #         # Tracking variables
    #         with torch.no_grad():
    #             for batch_input_ids, batch_attention_mask, batch_labels, batch_fea1, batch_fea2 in val_loader:
    #                 batch_input_ids = batch_input_ids.to(device)
    #                 batch_attention_mask = batch_attention_mask.to(device)
    #                 batch_fea1 = batch_fea1.to(device)
    #                 batch_fea2 = batch_fea2.to(device)
    #                 batch_labels = batch_labels.float().to(device).view(-1)  # (B,)
    #
    #                 logits, representation, fused, loss_contrast = model(
    #                     batch_fea1, batch_fea2, batch_input_ids, batch_attention_mask, labels=None
    #                 )
    #                 probs = torch.sigmoid(logits)  # (B,)
    #
    #                 batch_probs = probs.cpu().numpy().reshape(-1)
    #                 batch_preds = (batch_probs >= 0.5).astype(int)
    #
    #                 val_predictions.extend(batch_preds.tolist())
    #                 val_probabilities.extend(batch_probs.tolist())
    #                 val_labels_epoch.extend(batch_labels.cpu().numpy().tolist())
    #
    #
    #         # for batch_input_ids, batch_attention_mask, batch_labels,batch_fea1,batch_fea2 in val_loader:
    #         #     with torch.no_grad():
    #         #         # outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #         #         # fx, presention, _, loss2 = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)
    #         #         logits, representation, fused, loss_contrast = model(batch_fea1, batch_fea2, batch_input_ids,
    #         #                                                              batch_attention_mask, labels=None)
    #         #         probs = torch.sigmoid(logits)
    #         #
    #         #         # outputs = outputs.logits
    #         #     # outputs=fx
    #         #     batch_probs = probs.cpu().numpy()
    #         #     batch_preds = (batch_probs >= 0.5).astype(int)
    #         #
    #         #     val_predictions.extend(batch_preds.tolist())
    #         #     val_probabilities.extend(batch_probs.tolist())
    #         #     val_labels.extend(batch_labels.detach().cpu().numpy())
    #
    #         # Calculate MCC, AUC
    #         val_mcc = matthews_corrcoef(val_labels_epoch, val_predictions)
    #         val_auc = roc_auc_score(val_labels_epoch, val_probabilities)
    #
    #
    #         TP = TN = FP = FN = 0
    #         for y_true, y_pred in zip(val_labels_epoch, val_predictions):
    #             if y_pred == 1 and y_true == 1:
    #                 TP += 1
    #             elif y_pred == 0 and y_true == 0:
    #                 TN += 1
    #             elif y_pred == 1 and y_true == 0:
    #                 FP += 1
    #             else:
    #                 FN += 1
    #
    #         val_sensitivity = TP / (TP + FN + 1e-8)
    #         val_specificity = TN / (TN + FP + 1e-8)
    #         val_accuracy = accuracy_score(val_labels_epoch, val_predictions)
    #
    #
    #         print(f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}")
    #         output_string = f"Validation Accuracy: {val_accuracy:.4f} | Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | Validation MCC: {val_mcc:.4f} | Validation AUC: {val_auc:.4f}\n"
    #
    #         with open("../Result/ST_output.txt", "a") as file:
    #             file.write(output_string)
    #         # ====== Early stopping & 记录最优模型 ======
    #         if val_accuracy > best_acc + 1e-4:  # 有提升
    #             best_acc = val_accuracy
    #             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    #             best_metrics = (val_accuracy, val_sensitivity, val_specificity, val_mcc, val_auc)
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= patience:
    #                 print(f"Early stopping at epoch {epoch}")
    #                 break
    #
    #         # accuracies.append(val_accuracy)
    #         # val_sensitivity_list.append(val_sensitivity)
    #         # val_specificity_list.append(val_specificity)
    #         # val_mcc_list.append(val_mcc)
    #         # val_auc_list.append(val_auc)
    #
    #
    #     # 折结束后，恢复到该折最优模型
    #     if best_state is not None:
    #         model.load_state_dict(best_state)
    #
    #     # 保存最终模型
    #     model_save = model.state_dict()
    #     torch.save(model_save, f'../Result/ST_MMM_model_{fold + 1}.pth')
    #
    #     if best_metrics is not None:
    #         acc, sens, spec, mcc, auc = best_metrics
    #         accuracies.append(acc)
    #         val_sensitivity_list.append(sens)
    #         val_specificity_list.append(spec)
    #         val_mcc_list.append(mcc)
    #         val_auc_list.append(auc)
    #         print(
    #             f"[Fold {fold + 1}] Best -> Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
    #     else:
    #         print(f"[Fold {fold + 1}] No best_metrics available.")
    #
    # print("Cross Validation Results:")
    # print(
    #     f"Average Accuracy: {np.mean(accuracies):.4f} | "
    #     f"Average Sensitivity: {np.mean(val_sensitivity_list):.4f} | "
    #     f"Average Specificity: {np.mean(val_specificity_list):.4f} | "
    #     f"Average MCC: {np.mean(val_mcc_list):.4f} | "
    #     f"Average AUC: {np.mean(val_auc_list):.4f}"
    # )
    # print("Cross Validation Results:")
    # print(
    #     f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies, ddof=1):.4f} | "
    #     f"Average Sensitivity: {np.mean(val_sensitivity_list):.4f} ± {np.std(val_sensitivity_list, ddof=1):.4f} | "
    #     f"Average Specificity: {np.mean(val_specificity_list):.4f} ± {np.std(val_specificity_list, ddof=1):.4f} | "
    #     f"Average MCC: {np.mean(val_mcc_list):.4f} ± {np.std(val_mcc_list, ddof=1):.4f} | "
    #     f"Average AUC: {np.mean(val_auc_list):.4f} ± {np.std(val_auc_list, ddof=1):.4f}"
    # )





    # =======================
    # Complete model training（全量数据最终训练）
    # =======================
    set_seed(42)
    num_epochs_final = 100
    batch_size = 32

    # 用全部数据构建 dataset
    full_dataset = TensorDataset(input_ids, attention_mask, labels, fea1, fea2)
    num_samples = len(full_dataset)

    # 从全部数据中再划一个小验证集用于 early stopping（例如 10%）
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


    lambda_contrast =0# 先只训主任务

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


    # early stopping 变量
    best_auc_full = 0.0
    best_state_full = None
    best_acc_full = 0.0
    best_mcc_full = 0.0
    best_epoch_full = -1
    patience = 10
    patience_counter = 0

    print("===== Start full-data training =====")
    for epoch in range(num_epochs_final):
        # ---------- 训练 ----------
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

        # ---------- 在小验证集上 early stopping ----------
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

        # early stopping：根据 AUC 选择最优模型
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


    # 训练完后，恢复到最优 AUC 对应的参数
    if best_state_full is not None:
        model.load_state_dict(best_state_full)
        print(
            f"[Full] Best model at epoch {best_epoch_full}: "
            f"AUC={best_auc_full:.4f}, ACC={best_acc_full:.4f}, MCC={best_mcc_full:.4f}"
        )
    else:
        print("[Full] No best model found.")

    # 保存最终模型
    model_save = model.state_dict()
    torch.save(model_save, '../Result/ST_MMM_model_TEST.pth')
    print("Saved final full-data model to ../Result/ST_MMM_model_TEST.pth")




