import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding
class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """

    def __init__(self, in_features, hidden_units, num_task):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values, 1, 2)  # transpose to make it suitable for matrix multiplication
        # print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights
class MultiplicationFusion(nn.Module):
    """
    乘法融合模块（适应维度不匹配）。
    - 将 context 扩展到 T=32，然后元素-wise 乘法。
    - 输出形状: [B, 32, fused_dim]。
    """
    def __init__(self, in_dim, fused_dim):
        super(MultiplicationFusion, self).__init__()
        self.proj = nn.Linear(in_dim, fused_dim)  # 可选投影

    def forward(self, context, bilstm):
        """
        输入:
        - context: [B, 1, D]
        - bilstm: [B, 32, D]
        输出: [B, 32, fused_dim]
        """
        # 扩展 context 到 [B, 32, D]
        context_exp = context.expand(-1, bilstm.size(1), -1)  # [B, 32, D]
        fused = context_exp * bilstm  # 元素-wise 乘法 [B, 32, D]
        fused = F.relu(self.proj(fused))  # 可选：投影 + 非线性
        return fused
class ESMFeatureEncoder(nn.Module):
    def __init__(self, esm_dim=1152, hidden_dim=256, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        # 1) 降维 + 归一化
        self.proj = nn.Linear(esm_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        # 2) 轻量 Transformer 编码结构特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,   # (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 3) 注意力池化，把 (B, L, D) → (B, D)
        self.attn_vec = nn.Linear(hidden_dim, 1)

    def forward(self, fea2, mask=None):
        """
        fea2: (B, L, esm_dim)
        mask: (B, L)，1 表示有效位，0 表示 padding（如果有可选）
        """
        x = self.proj(fea2)           # (B, L, hidden_dim)
        x = self.ln(x)

        if mask is not None:
            # Transformer 的 key_padding_mask: True 表示要mask掉的位置
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, hidden_dim)

        # 注意力池化
        attn_scores = self.attn_vec(x).squeeze(-1)   # (B, L)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)      # (B, L)
        s = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, hidden_dim)

        return s, attn_weights   # s 就是“结构表征”
class GatedFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_seq, h_esm):
        gate = self.gate(torch.cat([h_seq, h_esm], dim=-1))
        return gate * h_seq + (1 - gate) * h_esm
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, dim=256):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             dim, num_heads=4, batch_first=True
#         )
#
#     def forward(self, seq_feat, esm_feat):
#         # Query = sequence, Key/Value = ESM
#         fused, attn = self.cross_attn(
#             query=seq_feat,
#             key=esm_feat,
#             value=esm_feat
#         )
#         return fused


class CrossAttentionFusion(nn.Module):
    """双向交叉注意力融合"""

    def __init__(self, dim, num_heads, dropout=0.0):
        super(CrossAttentionFusion, self).__init__()

        self.attn_cnn_to_esm = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_esm_to_cnn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: (B, 1, dim) - 通常是CNN特征
        # key/value: (B, 1, dim) - 通常是ESM特征

        # CNN到ESM的注意力
        cnn_enhanced, _ = self.attn_cnn_to_esm(query, key, value)

        # ESM到CNN的注意力
        esm_enhanced, _ = self.attn_esm_to_cnn(key, query, query)

        # 融合
        fused = (cnn_enhanced + esm_enhanced) / 2
        fused = self.norm(fused + self.dropout(fused))

        return fused
class Adapt_emb_CNNLSTM_ATT(nn.Module):
    def __init__(self):
        super(Adapt_emb_CNNLSTM_ATT, self).__init__()
        kernel_size = 8
        max_len = 42
        d_model = 33
        vocab_size = 28
        self.embedding = Embedding(vocab_size, d_model, max_len)
        # ---------------->>>
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=33,  # 32,      # input height
                out_channels=256,  # n_filters
                kernel_size=kernel_size),
            # dilation =2),     #!!!
            # padding = int(kernel_size/2)),
            # padding=(kernel_size-1)/2
            nn.ReLU(),  # activation
            # nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(256),
            nn.Dropout())
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=33,
        #         out_channels=256,
        #         kernel_size=kernel_size,
        #         padding=kernel_size // 2  # ✅ 保留边界信息
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3)
        # )
        self.Attention = BahdanauAttention(in_features=256, hidden_units=64, num_task=1)
        self.fc_task = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.fc_task1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # self.uu = nn.Linear(128, 2)
        self.classifier = nn.Linear(128, 1)

        self.linear = nn.Linear(35, 32 * 35)
        self.linear1 = nn.Linear(256, 80)

        self.fusion = MultiplicationFusion(in_dim=256, fused_dim=256)  # 84
        # self.fusion1 = CrossAttentionFusion(dim=256, num_heads=4, dropout=0.0)
        # self.fusion = AttentionFusion(in_dim=256, fused_dim=256)
        self.norm = nn.LayerNorm(256)

        self.fusion_gate = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出在 (0,1)，作为自适应权重
        )
        self.esm_encoder = ESMFeatureEncoder(esm_dim=1152, hidden_dim=256)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        # self.fusion_attn = nn.MultiheadAttention(
        #     embed_dim=256,
        #     num_heads=8,
        #     batch_first=True
        # )
        #
        # self.fusion_norm = nn.LayerNorm(256)


    def contrastive_loss(self, z1, z2, labels=None, temperature=0.2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim = torch.matmul(z1, z2.T) / temperature

        if labels is not None:
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(z1.device)
        else:
            mask = torch.eye(sim.size(0)).to(z1.device)

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return loss.mean()

    def info_nce_loss(z1, z2, temperature=0.2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        return F.cross_entropy(logits, labels)

    def compute_contrastive_loss(self, z1, z2, temperature=0.2):
        """改进的对比损失"""
        if temperature is None:
            temperature = self.config.temperature

        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / temperature

        # 对角线是正样本对
        labels = torch.arange(z1.size(0)).to(z1.device)

        # InfoNCE损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, x,fea2,batch_input_ids, batch_attention_mask, labels):
        # outputs = self.esm2(batch_input_ids, batch_attention_mask, return_dict=True)
        # token_embeddings = outputs.last_hidden_state
        umap = x.view(x.size(0), -1)
        x_first35 = x[:, :35]
        # x_last = x[:, 35:]
        # x_last = x_last.float()

        x_first35 = self.embedding(x_first35)#35*33
        x_first35 = x_first35.transpose(1, 2)  # 256*33*35
        batch_size, features, seq_len = x_first35.size()
        x = self.conv1(x_first35)
        x = x.transpose(1, 2)  # 256*28*256
        out = x  # (B, T, 256)
        # 全局 max-pooling 或 mean-pooling 都可以，先用 max 试试
        h_n = out.max(dim=1).values  # (B, 256)
        # h_n = self.linear1(h_n)
        context_vector, attention_weights = self.Attention(h_n, out)  # b*1*256 b*28*1 b*256 b*28*256
        # context_vector=h_n.unsqueeze(1)



        s, esm_attn = self.esm_encoder(fea2)  # s: (B, 256)
        h_n1=s
        s = s.unsqueeze(1)
        # fused=context_vector
        fused = self.fusion(context_vector, s)

        # fusion_input = torch.cat([context_vector, s], dim=-1)  # (B, 1, 512)
        # alpha = self.fusion_gate(fusion_input)  # (B, 1, 1)，自适应权重
        # # alpha 越大，越偏向 context_vector；越小，越偏向 s
        # fused = alpha * context_vector + (1.0 - alpha) * s

        # fused = self.fusion1(context_vector, s,s)
        # fused =torch.cat([context_vector, s], dim=-1)

        fused = torch.mean(fused, 1)
        representation = self.fc(fused)

        logits = self.classifier(representation).squeeze(-1)  # (B,) raw logits

        # 5) 对比损失（建议先做归一化）
        # h_n_norm = F.normalize(h_n, dim=-1)
        # h_n1_norm = F.normalize(h_n1, dim=-1)
        # contrastive = self.contrastive_loss(h_n_norm, h_n1_norm)
        # loss_contrast = self.contrastive_loss(
        #     context_vector.squeeze(1),
        #     s.squeeze(1),
        #     labels=labels
        # ) if labels is not None else torch.tensor(0.0, device=logits.device)
        loss_contrast=0
        # loss_contrast =self.compute_contrastive_loss(
        #     context_vector.squeeze(1),
        #     s.squeeze(1))

        return logits, representation, umap, esm_attn