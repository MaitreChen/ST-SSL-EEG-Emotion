import torch
import numpy as np


def mixup_data(x, y, alpha=0.2):
    """
    对 EEG 脑电波进行 Mixup 数据增强
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    计算 Mixup 后的混合交叉熵损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)