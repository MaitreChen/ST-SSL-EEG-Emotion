import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


# ----------------- 1. 梯度反转层 (GRL) -----------------
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 正向传播：记录当前的 alpha 值，特征 x 原样通过
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：将回传的梯度乘以 -alpha，实现对抗学习
        # 域判别器想让 loss 变小，乘以负号后，基座网络就会朝着让 loss 变大的方向更新
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ----------------- 2. 多尺度域判别器 -----------------
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[64, 32]):
        """
        EmoTeam - 跨群体对抗域判别器
        :param input_dim: Transformer CLS Token 的维度 (默认 128)
        """
        super().__init__()

        # 使用 MLP 逐级降维提取域特征
        layers = []
        in_d = input_dim
        for out_d in hidden_dims:
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            in_d = out_d

        # 最后一层输出 2 个类别：0 -> HC (健康), 1 -> MDD (抑郁)
        layers.append(nn.Linear(in_d, 2))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x, alpha=1.0):
        """
        :param x: 全局情感特征 (Batch, Embed_Dim)
        :param alpha: 梯度反转的强度参数
        """
        # 1. 经过梯度反转层
        reversed_x = grad_reverse(x, alpha)
        # 2. 判别器分类
        domain_logits = self.discriminator(reversed_x)
        return domain_logits


# ----------------- 3. 训练技巧：Alpha 动态调度器 -----------------
# 在对抗训练初期，判别器还很弱，如果直接给很强的反向梯度，会让基座网络崩溃。
# 因此 alpha 需要随着训练的进行，从 0 慢慢平滑过渡到 1。
def get_alpha(current_step, max_steps):
    """
    计算动态 alpha 值 (公式参考 DANN 论文)
    """
    p = current_step / max_steps
    # 使用 sigmoid 形状的曲线平滑递增
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    return float(alpha)


# ----------------- 独立测试区 -----------------
if __name__ == "__main__":
    # 模拟从 Transformer 吐出来的 [CLS] 特征 (Batch=16, Embed_Dim=128)
    dummy_cls_features = torch.randn(16, 128)

    # 模拟标签：假设前8个是 HC(0)，后8个是 MDD(1)
    # 这正是我们在 data_loader.py 里提取的 group 信息
    dummy_domain_labels = torch.tensor([0] * 8 + [1] * 8)

    # 初始化判别器
    domain_net = DomainDiscriminator(input_dim=128)

    print("--- 多尺度对抗域判别器测试 ---")

    # 测试在训练初期的表现 (alpha 很小)
    alpha_early = get_alpha(current_step=10, max_steps=1000)
    logits_early = domain_net(dummy_cls_features, alpha_early)
    print(f"训练初期 Alpha: {alpha_early:.4f}")
    print(f"域分类预测维度: {logits_early.shape}")  # 预期 [16, 2]

    # 计算 Domain Loss
    criterion = nn.CrossEntropyLoss()
    loss_domain = criterion(logits_early, dummy_domain_labels)
    print(f"模拟 Domain Loss: {loss_domain.item():.4f}")