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
#                 padding=kernel_size // 2  # ✅ 保留边界信息
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
#             nn.Sigmoid()  # 输出在 (0,1)，作为自适应权重
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
#         # out = x  # (B, T, 256)
#         # # 全局 max-pooling 或 mean-pooling 都可以，先用 max 试试
#         # h_n = out.max(dim=1).values  # (B, 256)
#         # # h_n = self.linear1(h_n)
#         # # h_n = self.linear1(h_n)
#         # context_vec, attention_weights = self.Attention(h_n, out)  # b*1*256 b*28*1 b*256 b*28*256
#
#         esm_vec, esm_attn = self.esm_encoder(fea2)  # s: (B, 256)
#         # h_n1=s
#         esm_vec = esm_vec.unsqueeze(1)
#
#
#         fusion_input = torch.cat([context_vec, esm_vec], dim=-1)
#         alpha = self.fusion_gate(fusion_input)  # (B, 1, 1)，自适应权重
#         # alpha 越大，越偏向 context_vector；越小，越偏向 s
#         fused = alpha * context_vec + (1.0 - alpha) * esm_vec
#         fused = fused.squeeze(1)
#         # fused = self.fusion(context_vector, s)
#         # fused = s
#         # ===== 4. 分类 =====
#         rep = self.fc(fused)
#         logits = self.classifier(rep).squeeze(-1)
#         # fused = torch.mean(fused, 1)
#         # representation = self.fc_task1(fused)
#         # logits = self.classifier(representation).squeeze(-1)  # (B,) raw logits
#
#         # 5) 对比损失（建议先做归一化）
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
# ====== 添加UMAP可视化函数 ======
def create_umap_visualization(representations, labels, probs, preds, title_suffix=""):
    """
    创建UMAP可视化

    参数:
        representations: numpy数组，形状 (n_samples, n_features)
        labels: 真实标签列表/数组
        probs: 预测概率列表/数组
        preds: 预测标签列表/数组
        title_suffix: 标题后缀
    """
    # 确保数据是numpy数组
    representations = np.array(representations)
    labels = np.array(labels).flatten()
    probs = np.array(probs).flatten()
    preds = np.array(preds).flatten()

    n_samples = len(representations)
    print(f"UMAP输入数据形状: {representations.shape}")
    print(f"样本数量: {n_samples}")
    print(f"标签分布: 0={np.sum(labels == 0)}, 1={np.sum(labels == 1)}")

    # 1. 标准化数据（重要！）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    representations_scaled = scaler.fit_transform(representations)

    # 2. UMAP降维
    print("正在运行UMAP降维...")

    # 创建UMAP降维器
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(50, n_samples - 1),  # 确保不超过样本数
        min_dist=0.5,
        metric='euclidean',
        random_state=42,
        n_epochs=500
    )

    # 执行降维
    embedding = reducer.fit_transform(representations_scaled)

    # 3. 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    palette = ['#1f77b4', '#ff7f0e']
    # 子图1: 按真实标签着色
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

    # 添加类别中心
    for label_val in [0, 1]:
        if np.any(labels == label_val):
            mask = labels == label_val
            center = embedding[mask].mean(axis=0)
            ax1.scatter(center[0], center[1], s=200, marker='*',
                        edgecolors='black', linewidth=2,
                        color=plt.cm.tab10(label_val), label=f'Class {label_val} center')

    # # 子图2: 按预测概率着色
    # ax2 = axes[0, 1]
    # scatter2 = ax2.scatter(
    #     embedding[:, 0], embedding[:, 1],
    #     c=probs, cmap='coolwarm',
    #     s=30, alpha=0.8, edgecolors='w', linewidth=0.5,
    #     vmin=0, vmax=1
    # )
    # ax2.set_title('Probability prediction heatmap\n(Blue: Low probability, Red: High probability)')
    # ax2.set_xlabel('UMAP 1')
    # ax2.set_ylabel('UMAP 2')
    # plt.colorbar(scatter2, ax=ax2, label='Predicted Probability')
    # ax2.grid(True, alpha=0.3)
    #
    # # 子图3: 预测正确性
    # ax3 = axes[0, 2]
    # correct = preds == labels
    # colors = ['#2E8B57' if c else '#DC143C' for c in correct]  # 绿色:正确, 红色:错误
    #
    # scatter3 = ax3.scatter(
    #     embedding[:, 0], embedding[:, 1],
    #     c=colors, s=30, alpha=0.8, edgecolors='w', linewidth=0.5
    # )
    #
    # correct_count = np.sum(correct)
    # error_count = np.sum(~correct)
    # accuracy = correct_count / n_samples * 100
    #
    # ax3.set_title(f'Predictive accuracy\ncorrect: {correct_count} ({accuracy:.1f}%), error: {error_count}')
    # ax3.set_xlabel('UMAP 1')
    # ax3.set_ylabel('UMAP 2')
    # ax3.grid(True, alpha=0.3)
    #
    # # 添加图例
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='#2E8B57', edgecolor='w', label=f'correct ({correct_count})'),
    #     Patch(facecolor='#DC143C', edgecolor='w', label=f'error ({error_count})')
    # ]
    # ax3.legend(handles=legend_elements, loc='upper right')
    #
    # # 子图4: 置信度分布
    # ax4 = axes[1, 0]
    # confidence = np.abs(probs - 0.5) * 2  # 映射到0-1
    #
    # if np.any(correct) and np.any(~correct):
    #     ax4.hist(confidence[correct], bins=20, alpha=0.7,
    #              label=f'correct (n={correct_count})', color='#dd81a6', density=True)
    #     ax4.hist(confidence[~correct], bins=20, alpha=0.7,
    #              label=f'error (n={error_count})', color='#ead2cc', density=True)
    #     ax4.legend()
    # else:
    #     ax4.hist(confidence, bins=20, alpha=0.7, color='blue', density=True)
    #
    # ax4.set_xlabel('Prediction confidence level')
    # ax4.set_ylabel('density')
    # ax4.set_title('Confidence distribution')
    # ax4.grid(True, alpha=0.3)
    #
    # # 子图5: 决策边界可视化
    # ax5 = axes[1, 1]
    #
    # # 根据概率大小着色
    # scatter5 = ax5.scatter(
    #     embedding[:, 0], embedding[:, 1],
    #     c=[palette[int(l)] for l in labels], cmap='tab10',
    #     s=30, alpha=0.8, edgecolors='w', linewidth=0.5,
    #     vmin=0, vmax=1
    # )
    #
    # # 尝试绘制决策边界
    # try:
    #     from sklearn.svm import SVC
    #     from matplotlib.colors import ListedColormap
    #
    #     # 训练SVM
    #     svm = SVC(kernel='rbf', C=1.0, probability=True)
    #     svm.fit(embedding, preds)
    #
    #     # 创建网格
    #     h = 0.02  # 网格步长
    #     x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
    #     y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                          np.arange(y_min, y_max, h))
    #
    #     # 预测网格点
    #     Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #
    #     # 绘制决策边界
    #     cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    #     ax5.contourf(xx, yy, Z, alpha=0.2, cmap=cmap_light)
    #
    #     ax5.set_title('Prediction label + Decision boundary')
    # except:
    #     ax5.set_title('Prediction label ')
    #
    # ax5.set_xlabel('UMAP 1')
    # ax5.set_ylabel('UMAP 2')
    # ax5.grid(True, alpha=0.3)
    #
    # # 子图6: 样本密度热图
    # ax6 = axes[1, 2]
    #
    # # 使用hexbin创建密度图
    # hb = ax6.hexbin(embedding[:, 0], embedding[:, 1],
    #                 gridsize=30, cmap='Blues', alpha=0.8)
    #
    # ax6.set_title('Sample density heatmap')
    # ax6.set_xlabel('UMAP 1')
    # ax6.set_ylabel('UMAP 2')
    # plt.colorbar(hb, ax=ax6, label='Sample density')
    # ax6.grid(True, alpha=0.3)

    # 设置总标题
    plt.suptitle(f'Representation UMAP visual {title_suffix}\n'
                 f'characteristic dimension: {representations.shape[1]}, sample size: {n_samples}',
                 fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()
    # fig.savefig('../Result/figures/st_front_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.02)

    return embedding


def create_simple_umap(representations, labels, title="UMAP Visualization"):
    """
    创建一个简单的UMAP可视化
    """
    representations = np.array(representations)
    labels = np.array(labels).flatten()

    # 标准化
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

    # 绘图
    plt.figure(figsize=(10, 8))

    # 根据标签着色
    colors = ['blue' if label == 0 else 'red' for label in labels]

    plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, s=40, alpha=0.7, edgecolors='w', linewidth=0.5
    )

    # 添加类别标签
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
    all_representations = []  # 新增：收集所有representation
    all_logits = []  # 新增：收集logits

    for batch_input_ids, batch_attention_mask, batch_labels,batch_fea1,batch_fea2 in test_loader:
        with torch.no_grad():
            # outputs = model(batch_input_ids, batch_attention_mask,
            #                 labels=batch_labels, return_dict=True)
            # fx, presention, _, loss2,esm_attn = model(batch_fea1,batch_fea2,batch_input_ids, batch_attention_mask, labels=batch_labels)
            logits, representation, fused, esm_attn = model(batch_fea1, batch_fea2, batch_input_ids,
                                                                 batch_attention_mask, labels=batch_labels)
            probs = torch.sigmoid(logits)

            # 收集representation和logits
            all_representations.append(fused.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

            # seq = test_data[0]
            # batch_idx = 0
            # # 模型输出：logits, rep, fused, loss2, esm_attn = model(...)
            # attn_full = esm_attn[batch_idx].detach().cpu()  # 形状: (35,)
            # # 去掉 CLS 和 EOS，只保留 33 个残基
            # attn_residue = attn_full[1:-1]  # 形状: (33,)
            # assert len(seq) == attn_residue.shape[0] == 33
            #
            # # Top-k 残基
            # k = 10
            # values, indices = torch.topk(attn_residue, k)
            # indices = indices.tolist()
            # values = values.tolist()
            #
            # print(f"Top-{k} 重要残基(33 个真实残基里):")
            # for rank, (i, w) in enumerate(zip(indices, values), start=1):
            #     aa = seq[i]  # 这里 i 是 0-based
            #     print(f"#{rank}: 位置 {i + 1}, 氨基酸 {aa}, 注意力权重 {w:.4f}")
            # import matplotlib.pyplot as plt
            #
            # attn_np = attn_residue.numpy()  # (33,)
            # positions = np.arange(1, len(seq) + 1)
            #
            # # 归一化到 [0, 1]
            # attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-12)
            #
            # plt.figure(figsize=(10, 3))
            # plt.stem(positions, attn_norm)
            # plt.xlabel("Residue index (1-based)")
            # plt.ylabel("Normalized attention")
            # plt.title("ESM attention (normalized)")
            # plt.xticks(positions, list(seq))
            # plt.tight_layout()
            # plt.show()

        # outputs = outputs.logits
        # outputs = logits
        # batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
        # batch_probs = outputs.cpu().detach().numpy()
        # all_preds.extend(batch_preds.tolist())
        # all_probs.extend(batch_probs.tolist())
        # all_labels.extend(batch_labels.detach().cpu().numpy())
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
    # # ====== 可视化混淆矩阵 ======
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Predicted 0', 'Predicted 1'],
    #             yticklabels=['Actual 0', 'Actual 1'])
    # plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.tight_layout()
    # plt.show()
    #
    # # ====== 可视化ROC曲线 ======
    # from sklearn.metrics import roc_curve
    #
    # fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    # plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate', fontsize=12)
    # plt.ylabel('True Positive Rate', fontsize=12)
    # plt.title('ROC Curve', fontsize=14)
    # plt.legend(loc='lower right')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    #
    # print("\n" + "=" * 60)
    # print("测试完成！")
    # print("=" * 60)
    # # ====== 准备UMAP数据 ======
    # # 堆叠所有batch的representation
    # if len(all_representations) > 0:
    #     representations_concat = np.vstack(all_representations)
    #     print(f"堆叠后的representation形状: {representations_concat.shape}")
    #
    #     # 检查数据一致性
    #     n_samples = len(representations_concat)
    #     print(f"样本总数: {n_samples}")
    #     print(f"标签数: {len(all_labels)}")
    #     print(f"概率数: {len(all_probs)}")
    #     print(f"预测数: {len(all_preds)}")
    #
    #     # 确保数据长度一致
    #     min_length = min(n_samples, len(all_labels), len(all_probs), len(all_preds))
    #
    #     if min_length < n_samples:
    #         print(f"警告: 数据长度不一致，截取到 {min_length} 个样本")
    #         representations_concat = representations_concat[:min_length]
    #         all_labels = all_labels[:min_length]
    #         all_probs = all_probs[:min_length]
    #         all_preds = all_preds[:min_length]
    #
    #     print(f"\nUMAP输入数据:")
    #     print(f"Representations: {representations_concat.shape}")
    #     print(f"Labels: {len(all_labels)}")
    #     print(f"Probabilities: {len(all_probs)}")
    #     print(f"Predictions: {len(all_preds)}")
    #
    #     # ====== 运行UMAP可视化 ======
    #     print("\n" + "=" * 60)
    #     print("开始UMAP可视化分析")
    #     print("=" * 60)
    #
    #     # 创建完整的UMAP可视化
    #     try:
    #         umap_embedding = create_umap_visualization(
    #             representations_concat,
    #             all_labels,
    #             all_probs,
    #             all_preds,
    #             title_suffix="(Test Set)"
    #         )
    #         print("✓ UMAP可视化完成")
    #     except Exception as e:
    #         print(f"UMAP可视化失败: {e}")
    #         print("尝试简单UMAP...")
    #
    #         # 尝试简单版本
    #         try:
    #             simple_embedding = create_simple_umap(
    #                 representations_concat,
    #                 all_labels,
    #                 title="Test Set Representation UMAP"
    #             )
    #             print("✓ 简单UMAP可视化完成")
    #         except Exception as e2:
    #             print(f"简单UMAP也失败: {e2}")
    #
    #     # ====== 额外分析：错误样本分析 ======
    #     print("\n" + "=" * 60)
    #     print("错误样本分析")
    #     print("=" * 60)
    #
    #     all_labels_array = np.array(all_labels).flatten()
    #     all_preds_array = np.array(all_preds).flatten()
    #     all_probs_array = np.array(all_probs).flatten()
    #
    #     correct_mask = all_preds_array == all_labels_array
    #     error_mask = ~correct_mask
    #
    #     error_count = np.sum(error_mask)
    #
    #     if error_count > 0:
    #         print(f"错误样本数: {error_count} ({error_count / len(all_labels_array) * 100:.1f}%)")
    #         print(f"错误样本的真实标签分布: 0={np.sum(all_labels_array[error_mask] == 0)}, "
    #               f"1={np.sum(all_labels_array[error_mask] == 1)}")
    #         print(f"错误样本的平均置信度: {np.mean(all_probs_array[error_mask]):.3f}")
    #
    #         # 可视化错误样本在UMAP中的位置
    #         if 'umap_embedding' in locals():
    #             plt.figure(figsize=(10, 8))
    #
    #             # 绘制所有样本
    #             plt.scatter(umap_embedding[correct_mask, 0], umap_embedding[correct_mask, 1],
    #                         c='lightgray', s=30, alpha=0.3, label='Correct sample')
    #
    #             # 突出显示错误样本
    #             error_colors = ['blue' if label == 0 else 'red'
    #                             for label in all_labels_array[error_mask]]
    #
    #             plt.scatter(umap_embedding[error_mask, 0], umap_embedding[error_mask, 1],
    #                         c=error_colors, s=50, alpha=0.8, edgecolors='black',
    #                         linewidth=1.5, label='Error sample ')
    #
    #             plt.xlabel('UMAP 1', fontsize=12)
    #             plt.ylabel('UMAP 2', fontsize=12)
    #             plt.title(f'Distribution of error samples (n={error_count})', fontsize=14)
    #             plt.legend()
    #             plt.grid(True, alpha=0.3)
    #             plt.tight_layout()
    #             plt.show()
    #     else:
    #         print("✓ 所有样本都预测正确！")
    #
    # else:
    #     print("警告: 没有收集到representation数据")

    # # Calculate metrics
    # mcc = matthews_corrcoef(all_labels, all_preds)
    # auc = roc_auc_score(all_labels, all_probs)
    # TP = TN = FP = FN = 0
    # for i in range(len(all_labels)):
    #     if all_preds[i] == 1 and all_labels[i] == 1:
    #         TP += 1
    #     elif all_preds[i] == 0 and all_labels[i] == 0:
    #         TN += 1
    #     elif all_preds[i] == 1 and all_labels[i] == 0:
    #         FP += 1
    #     else:
    #         FN += 1
    # acc = (TP + TN) / (TP + TN + FP + FN)
    # sn = TP / (TP + FN)
    # sp = TN / (TN + FP)
    # print(f"Test Acc: {acc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
