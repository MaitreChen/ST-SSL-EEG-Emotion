import torch
import torch.nn as nn


class EEGMaskedAutoencoder(nn.Module):
    def __init__(self, backbone, mask_ratio=0.4):
        """
        EmoTeam - 频域自监督掩码自编码器 (FFT-MAE)
        创新点：放弃预测混沌的时域波形，转而预测频域的对数振幅谱
        """
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_size
        self.in_channels = backbone.patch_embed.in_channels

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 🚨 核心改动 1：计算 FFT 后的特征维度
        # 对于长度为 50 的实数序列，rfft 输出的复数频点个数为 (50 // 2) + 1 = 26
        self.fft_size = (self.patch_size // 2) + 1

        # 解码器输出维度从 1500 改为预测频谱特征：30个通道 * 26个频点 = 780
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.in_channels * self.fft_size)
        )

    def random_masking(self, x_tokens):
        # ... (此处与之前完全一致，保留原逻辑) ...
        B, L, D = x_tokens.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, L, device=x_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_kept = torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask_tokens = self.mask_token.expand(B, L - len_keep, -1)
        x_masked_combined = torch.cat([x_kept, mask_tokens], dim=1)
        x_restored = torch.gather(x_masked_combined, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([B, L], device=x_tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_restored, mask

    def forward(self, x):
        B = x.shape[0]

        # 1. 经过特征基座
        patches = self.backbone.patch_embed(x).transpose(1, 2)
        masked_patches, mask = self.random_masking(patches)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x_encoded = torch.cat((cls_tokens, masked_patches), dim=1)
        x_encoded = self.backbone.pos_embed(x_encoded)
        x_encoded = self.backbone.transformer(x_encoded)

        reconstructed_tokens = x_encoded[:, 1:, :]

        # 2. 预测频域特征 (B, Num_Patches, 780)
        pred_fft = self.decoder(reconstructed_tokens)

        # 生成真实的频域目标 (FFT Target)
        L = patches.shape[1]
        # 提取出原始的 Patch 波形: (B, L, 30, 50)
        patches_raw = x.view(B, self.in_channels, L, self.patch_size).permute(0, 2, 1, 3)

        # 沿时间维度进行快速傅里叶变换，求绝对值得到振幅谱: (B, L, 30, 26)
        fft_target = torch.abs(torch.fft.rfft(patches_raw, dim=-1))

        # 取对数压缩动态范围 (加 1e-6 防止 log(0))，这是脑电特征工程的标准操作
        fft_target = torch.log(fft_target + 1e-6)

        # 展平为与 pred_fft 对应的形状: (B, L, 780)
        x_target = fft_target.contiguous().view(B, L, -1)

        # 局部频点标准化 (Patch-wise Norm)，让模型关注频谱的相对分布形状
        target_mean = x_target.mean(dim=-1, keepdim=True)
        target_var = x_target.var(dim=-1, keepdim=True)
        x_target = (x_target - target_mean) / (target_var + 1e-6) ** 0.5

        return pred_fft, x_target, mask