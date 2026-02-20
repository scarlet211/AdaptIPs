import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, EsmForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import TensorDataset
import pickle
import torch.nn.functional as F
from transformers import EsmModel
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
import numpy as np
from model1 import Adapt_emb_CNNLSTM_ATT
# from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

import seaborn as sns
import umap
# import umap.plot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Adapt_emb_CNNLSTM_ATT(nn.Module):
#     def __init__(self):
#         super(Adapt_emb_CNNLSTM_ATT, self).__init__()
#         kernel_size = 8
#         max_len = 42
#         d_model = 33
#         vocab_size = 28
#         self.embedding = Embedding(vocab_size, d_model, max_len)
#         # ---------------->>>
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv1d(
#         #         in_channels=33,  # 32,      # input height
#         #         out_channels=256,  # n_filters
#         #         kernel_size=kernel_size),
#         #     # dilation =2),     #!!!
#         #     # padding = int(kernel_size/2)),
#         #     # padding=(kernel_size-1)/2
#         #     nn.ReLU(),  # activation
#         #     # nn.MaxPool1d(kernel_size=2),
#         #     nn.BatchNorm1d(256),
#         #     nn.Dropout())
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=33,
#                 out_channels=256,
#                 kernel_size=kernel_size,
#                 padding=kernel_size // 2  
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3)
#         )
#         self.attention = BahdanauAttention(in_features=256, hidden_units=64, num_task=1)
#         self.fc_task = nn.Sequential(
#             nn.Linear(256, 32),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             # nn.Linear(64, 32),
#             # nn.Dropout(0.5),
#             # nn.ReLU(),
#             nn.Linear(32, 2),
#         )
#         self.fc_task1 = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
#
#         self.classifier = nn.Linear(128, 1)
#         self.linear = nn.Linear(35, 32 * 35)
#         self.linear1 = nn.Linear(256, 80)
#
#         self.fusion = MultiplicationFusion(in_dim=256, fused_dim=256)  # 84
#         # self.fusion = AttentionFusion(in_dim=256, fused_dim=256)
#         self.norm = nn.LayerNorm(256)
#
#         self.fusion_gate = nn.Sequential(
#             nn.Linear(256 * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  
#         )
#         self.esm_encoder = ESMFeatureEncoder(esm_dim=1152, hidden_dim=256)
#
#     # def contrastive_loss(self, proj1, proj2, label=None):
#     #     proj1 = F.normalize(proj1, dim=1)
#     #     proj2 = F.normalize(proj2, dim=1)
#     #     dot = torch.matmul(proj1, proj2.T) / 1
#     #     dot_max, _ = torch.max(dot, dim=1, keepdim=True)
#     #     dot = dot - dot_max.detach()
#     #
#     #     exp_dot = torch.exp(dot)
#     #     log_prob = torch.diag(dot, 0) - torch.log(exp_dot.sum(1))
#     #     cont_loss = -log_prob.mean()
#     #     return cont_loss
#
#     def contrastive_loss(self, z1, z2, labels=None, temperature=0.2):
#         z1 = F.normalize(z1, dim=1)
#         z2 = F.normalize(z2, dim=1)
#
#         sim = torch.matmul(z1, z2.T) / temperature
#
#         if labels is not None:
#             labels = labels.view(-1, 1)
#             mask = torch.eq(labels, labels.T).float().to(z1.device)
#         else:
#             mask = torch.eye(sim.size(0)).to(z1.device)
#
#         exp_sim = torch.exp(sim)
#         log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
#
#         return loss.mean()
#
#     def forward(self, x,fea2,batch_input_ids, batch_attention_mask, labels):
#         # outputs = self.esm2(batch_input_ids, batch_attention_mask, return_dict=True)
#         # token_embeddings = outputs.last_hidden_state
#
#         x_first35 = x[:, :35]
#         x_first35 = self.embedding(x_first35)#35*33
#         x_first35 = x_first35.transpose(1, 2)  # 256*33*35
#         batch_size, features, seq_len = x_first35.size()
#         x = self.conv1(x_first35)
#         x = x.transpose(1, 2)  # 256*28*256
#         query = x.mean(dim=1, keepdim=True)
#         context_vec, _ = self.attention(query, x)
#
#         
#
#         esm_vec, esm_attn = self.esm_encoder(fea2)  # s: (B, 256)
#         # h_n1=s
#         esm_vec = esm_vec.unsqueeze(1)
#
#
#         fusion_input = torch.cat([context_vec, esm_vec], dim=-1)
#         alpha = self.fusion_gate(fusion_input)  # (B, 1, 1)，Adaptive weights
#        
#         fused = alpha * context_vec + (1.0 - alpha) * esm_vec
#         fused = fused.squeeze(1)
#         # fused = self.fusion(context_vector, s)
#         # fused = s
#         # ===== 4. classifier =====
#         rep = self.fc(fused)
#         logits = self.classifier(rep).squeeze(-1)
#         # fused = torch.mean(fused, 1)
#         # representation = self.fc_task1(fused)
#         # logits = self.classifier(representation).squeeze(-1)  # (B,) raw logits
#
#         # 5) loss
#         # h_n_norm = F.normalize(h_n, dim=-1)
#         # h_n1_norm = F.normalize(h_n1, dim=-1)
#         # contrastive = self.contrastive_loss1(h_n_norm, h_n1_norm,labels)
#         loss_contrast = self.contrastive_loss(
#             context_vec.squeeze(1),
#             esm_vec.squeeze(1),
#             labels=labels
#         ) if labels is not None else torch.tensor(0.0, device=logits.device)
#
#         return logits, rep, fused, loss_contrast

def create_umap_visualization(representations, labels, probs, preds, title_suffix=""):
    """
    Create UMAP visualization

  
    """
    
    representations = np.array(representations)
    labels = np.array(labels).flatten()
    probs = np.array(probs).flatten()
    preds = np.array(preds).flatten()

    n_samples = len(representations)
    print(f"Shape of the input data for UMAP: {representations.shape}")
    print(f"sample size: {n_samples}")
    print(f"Label Distribution: 0={np.sum(labels == 0)}, 1={np.sum(labels == 1)}")

    # 1. standardized data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    representations_scaled = scaler.fit_transform(representations)

    
  

    # Create an UMAP dimensionality reducer
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(50, n_samples - 1),  
        min_dist=0.5,
        metric='euclidean',
        random_state=42,
        n_epochs=500
    )

    # Perform dimensionality reduction
    embedding = reducer.fit_transform(representations_scaled)

    # 3. Create visual charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    palette = ['#1f77b4', '#ff7f0e']
    
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        embedding[:, 0], embedding[:, 1],
        c=[palette[int(l)] for l in labels], cmap='tab10',
        s=30, alpha=0.8, edgecolors='w', linewidth=0.5,
        vmin=0, vmax=1
    )
    ax1.set_title(f'Actual label distribution(S/T)\n0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.grid(True, alpha=0.3)

    # Add category center
    for label_val in [0, 1]:
        if np.any(labels == label_val):
            mask = labels == label_val
            center = embedding[mask].mean(axis=0)
            ax1.scatter(center[0], center[1], s=200, marker='*',
                        edgecolors='black', linewidth=2,
                        color=plt.cm.tab10(label_val), label=f'Class {label_val} center')
    
    plt.suptitle(f'Representation UMAP visual {title_suffix}\n'
                 f'characteristic dimension: {representations.shape[1]}, sample size: {n_samples}',
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()
    # fig.savefig('../Result/figures/st_front_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.02)

    return embedding


def create_simple_umap(representations, labels, title="UMAP Visualization"):
    """
    Create a simple UMAP visualization
    """
    representations = np.array(representations)
    labels = np.array(labels).flatten()

    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    representations_scaled = scaler.fit_transform(representations)

    # UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(representations_scaled) - 1),
        min_dist=0.1,
        random_state=42
    )

    embedding = reducer.fit_transform(representations_scaled)

    # plot
    plt.figure(figsize=(10, 8))

    # According to the label coloring
    colors = ['blue' if label == 0 else 'red' for label in labels]

    plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, s=40, alpha=0.7, edgecolors='w', linewidth=0.5
    )

    # Add category tags
    for label_val in [0, 1]:
        mask = labels == label_val
        if np.any(mask):
            center = embedding[mask].mean(axis=0)
            plt.scatter(center[0], center[1], s=300, marker='*',
                        edgecolors='black', linewidth=2,
                        color='blue' if label_val == 0 else 'red',
                        label=f'Class {label_val}')

    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title(f'{title}\n(n={len(representations)}, features={representations.shape[1]})',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return embedding


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
def test_model(test_data, test_label):

    # Device setup
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("../ESM2-150M")
    # model = EsmForSequenceClassification.from_pretrained("../ESM2-150M",
    #                                                     num_labels=1).to(device)
    model = Adapt_emb_CNNLSTM_ATT().to(device)

    # Load the best saved model
    model.load_state_dict(torch.load('../Result/ST_MMM_model_TEST.pth', map_location='cuda'))
    model.eval()

    # best_thr = float(np.load("../Result/best_threshold.npy"))
    # print(f"✅ Using optimal threshold: {best_thr:.2f}")
    # Data preprocessing
    encoded_texts = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    token_list, max_len = transform_token2index(test_data)
    fea = make_data_with_unified_length(token_list, max_len)
    fea1 = torch.tensor(fea).to(device)

    # esmcmodel = ESMC.from_pretrained("esmc_600m")
    # embedding = []
    # count=0
    # res = [''.join(sub) for sub in test_data]
    # for i in res:
    #     te = esmcmain(esmcmodel, i).squeeze().cpu().detach().numpy()
    #     embedding.append(te)
    #     count += 1
    #     print(count)
    # fea2 = np.array(embedding)
    # np.save("test-fea2.npy", fea2)
    fea2 = np.load("test-fea2.npy")
    print(fea2.shape)
    fea2 = torch.tensor(fea2).to(device)

    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    test_dataset = TensorDataset(input_ids, attention_mask, labels,fea1,fea2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Inference on the test data
    all_preds = []
    all_probs = []
    all_labels = []
    all_representations = []  
    all_logits = []  

    for batch_input_ids, batch_attention_mask, batch_labels,batch_fea1,batch_fea2 in test_loader:
        with torch.no_grad():
            # outputs = model(batch_input_ids, batch_attention_mask,
            #                 labels=batch_labels, return_dict=True)
            # fx, presention, _, loss2,esm_attn = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)
            logits, representation, fused, esm_attn = model(batch_fea1, batch_fea2, batch_input_ids,
                                                                 batch_attention_mask, labels=batch_labels)
            probs = torch.sigmoid(logits)

            # collect representation and logits
            all_representations.append(fused.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

        
        batch_probs = probs.cpu().numpy()
        batch_preds = (batch_probs >= 0.5).astype(int)

        all_preds.extend(batch_preds.tolist())
        all_probs.extend(batch_probs.tolist())
        all_labels.extend(batch_labels.detach().cpu().numpy())

    # ===== Metrics =====
    all_labels = np.array(all_labels).reshape(-1)
    all_preds = np.array(all_preds).reshape(-1)
    all_probs = np.array(all_probs).reshape(-1)

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN + 1e-8)
    sp = TN / (TN + FP + 1e-8)

    print(
        f"✅ Test Results | "
        f"Acc={acc:.4f} | "
        f"SN={sn:.4f} | "
        f"SP={sp:.4f} | "
        f"MCC={mcc:.4f} | "
        f"AUC={auc:.4f}"
    )
 
