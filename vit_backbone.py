import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码：天生支持变长序列输入
    """

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        # 创建一个足够长的位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # 如果输入的序列长度居然超过了 max_len，抛出明确警告
        if seq_len > self.pe.size(0):
            raise ValueError(f"序列长度 ({seq_len}) 超出了位置编码的最大容量 ({self.pe.size(0)})")
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class EEGTemporalViT(nn.Module):
    def __init__(self, in_channels=30, patch_size=50, embed_dim=256, num_heads=8, num_layers=4):
        """
        EmoTeam - 时空联合掩码 Transformer 基座
        :param in_channels: 脑电通道数 (30)
        :param patch_size: 每个时间块的长度 (250采样点 = 1秒)
        :param embed_dim: 隐层维度
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. 核心创新点：使用一维卷积实现无重叠的时间分块 (Patching)
        # 输入 (Batch, 30, Length) -> 输出 (Batch, Embed_Dim, Num_Patches)
        self.patch_embed = nn.Conv1d(in_channels=in_channels,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        # 2. 全局 [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. 变长位置编码
        self.pos_embed = PositionalEncoding(d_model=embed_dim, max_len=2000)

        # 4. Transformer 编码器层
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
        #                                            nhead=num_heads,
        #                                            dim_feedforward=embed_dim * 4,
        #                                            batch_first=True,
        #                                            dropout=0.1)

        # 在 vit_backbone.py 中修改：
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   batch_first=True,
                                                   dropout=0.1,
                                                   norm_first=True)  # 🚨 救命神仙参数：开启 Pre-LN

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        :param x: 脑电信号输入 (Batch, 30, Length)
        :return: cls_output (Batch, Embed_Dim), all_tokens (Batch, Seq_Len, Embed_Dim)
        """
        B = x.shape[0]

        # 阶段 A：分块与嵌入
        # x shape: (B, 30, L) -> (B, Embed_Dim, Num_Patches)
        x = self.patch_embed(x)

        # 维度转换以适配 Transformer: (B, Num_Patches, Embed_Dim)
        x = x.transpose(1, 2)

        # 阶段 B：拼接 [CLS] Token
        # 扩展 cls_token 以匹配 Batch Size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, Num_Patches + 1, Embed_Dim)

        # 阶段 C：注入位置信息
        x = self.pos_embed(x)

        # 阶段 D：Transformer 特征提取
        x = self.transformer(x)

        # 提取 [CLS] Token 作为全局情感表征，同时返回所有 Token 以备自监督掩码重建使用
        cls_output = x[:, 0, :]
        all_tokens = x

        return cls_output, all_tokens


# ----------------- 独立压力测试 -----------------
if __name__ == "__main__":
    # 模拟构建模型
    model = EEGTemporalViT(in_channels=30, patch_size=250, embed_dim=128)

    print("--- 压力测试：变长序列无缝推理 ---")

    # 1. 模拟训练集数据 (50秒 = 12500 采样点)
    train_dummy = torch.randn(16, 30, 12500)  # Batch=16
    cls_out_train, all_tokens_train = model(train_dummy)
    print(f"训练集输入 -> {train_dummy.shape}")
    print(f"  > CLS 输出特征维度: {cls_out_train.shape}")
    # 预期 token 数量: 1个CLS + (12500/250)个Patch = 51
    print(f"  > 所有 Token 维度: {all_tokens_train.shape}\n")

    # 2. 模拟测试集数据 (10秒 = 2500 采样点)
    test_dummy = torch.randn(16, 30, 2500)
    cls_out_test, all_tokens_test = model(test_dummy)
    print(f"测试集输入 -> {test_dummy.shape}")
    print(f"  > CLS 输出特征维度: {cls_out_test.shape}")
    # 预期 token 数量: 1个CLS + (2500/250)个Patch = 11
    print(f"  > 所有 Token 维度: {all_tokens_test.shape}")