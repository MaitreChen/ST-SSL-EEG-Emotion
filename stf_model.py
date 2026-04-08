# stf_model.py
import torch
import torch.nn as nn
import math
from torch.autograd import Function


# --- 1. 梯度反转层 (GRL) ---
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# --- 2. 时-空-频 特征提取块 ---
class SpatioTemporalFreqBlock(nn.Module):
    def __init__(self, in_channels=30, embed_dim=128):
        super().__init__()
        # 1. 空域滤波器 (Spatial Filter)：跨 30 个电极融合
        self.spatial_conv = nn.Conv2d(1, embed_dim, kernel_size=(in_channels, 1))
        self.bn_spatial = nn.BatchNorm2d(embed_dim)

        # 2. 多尺度频域/时域滤波器 (Multi-scale Temporal/Freq Filter)
        # 不同的 kernel_size 相当于提取不同频率段的脑电波（Delta, Theta, Alpha, Beta）
        self.freq_conv_high = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 15), padding=(0, 7))
        self.freq_conv_mid = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 31), padding=(0, 15))
        self.freq_conv_low = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 63), padding=(0, 31))

        self.bn_temporal = nn.BatchNorm2d(embed_dim * 3)
        self.elu = nn.ELU()

        # 降维回 embed_dim
        self.project = nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 30, Length)
        x = x.unsqueeze(1)  # (B, 1, 30, L)

        # 空域融合
        x = self.spatial_conv(x)  # (B, embed_dim, 1, L)
        x = self.elu(self.bn_spatial(x))

        # 多尺度时/频特征提取
        x_high = self.freq_conv_high(x)
        x_mid = self.freq_conv_mid(x)
        x_low = self.freq_conv_low(x)

        # 拼接多尺度特征
        x_multi = torch.cat([x_high, x_mid, x_low], dim=1)  # (B, embed_dim*3, 1, L)
        x_multi = self.elu(self.bn_temporal(x_multi)).squeeze(2)  # (B, embed_dim*3, L)

        # 通道降维
        x_out = self.project(x_multi)  # (B, embed_dim, L)
        return x_out


# --- 3. 端到端统一框架 ---
class EEGEmoSTFNetwork(nn.Module):
    def __init__(self, in_channels=30, patch_size=50, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()

        # A. 时空频联合提取模块
        self.stf_block = SpatioTemporalFreqBlock(in_channels, embed_dim)

        # B. 序列分块 (Patching) - 在特征提取后进行
        self.patch_embed = nn.Conv1d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

        # C. Transformer 编码器
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 2000, embed_dim))  # 简化为可学习的参数位置编码

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4, batch_first=True,
                                                   dropout=0.3, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # D. 情绪分类头
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

        # E. 群体/域判别器 (HC vs MDD)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=0.0):
        B = x.shape[0]

        # 1. 提取 STF 局部特征
        x = self.stf_block(x)  # (B, embed_dim, L)

        # 2. 转换为 Patch Tokens
        x = self.patch_embed(x).transpose(1, 2)  # (B, Num_Patches, embed_dim)

        # 3. 拼接 CLS 和 位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        # 4. Transformer 全局建模
        x = self.transformer(x)
        cls_feat = x[:, 0, :]

        # 5. 分支输出
        emo_logits = self.emotion_classifier(cls_feat)

        reversed_feat = GradientReversal.apply(cls_feat, alpha)
        dom_logits = self.domain_discriminator(reversed_feat)

        return emo_logits, dom_logits