import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from data_loader import EEGEmoDataset
from stf_model import EEGEmoSTFNetwork


# Mixup 数据增强函数
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 动态 Alpha 调度 (用于 GRL)
def get_alpha(current_step, max_steps):
    p = current_step / max_steps
    return float(2. / (1. + np.exp(-10 * p)) - 1)


def train_loso_cv():
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 3e-4
    DATA_ROOT = './data'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 注意：此时 Dataloader 会自动把训练集裁剪为 2500 点
    full_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train', crop_len=2500)
    all_subjects = list(set([sample['subject_id'] for sample in full_dataset.samples]))
    all_subjects.sort()

    fold_results = []

    # 比赛时去掉 [:5]，跑全量被试
    for fold, val_subject in enumerate(all_subjects):
        print(f"\n{'=' * 50}\n🚀 Fold {fold + 1}/{len(all_subjects)} | Unseen Subject: {val_subject}\n{'=' * 50}")

        train_indices = [i for i, sample in enumerate(full_dataset.samples) if sample['subject_id'] != val_subject]
        val_indices = [i for i, sample in enumerate(full_dataset.samples) if sample['subject_id'] == val_subject]

        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)

        # 实例化 STF 端到端模型
        model = EEGEmoSTFNetwork(embed_dim=128, num_layers=4).to(device)

        # 使用余弦退火学习率
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        criterion = nn.CrossEntropyLoss()

        total_steps = EPOCHS * len(train_loader)
        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            current_step = epoch * len(train_loader)
            train_loss = 0.0

            for batch_idx, (data, emo_labels, _, group_labels, _) in enumerate(train_loader):
                data, emo_labels = data.to(device), emo_labels.to(device)
                dom_labels = torch.tensor([0 if g == 'HC' else 1 for g in group_labels]).to(device)

                alpha = get_alpha(current_step + batch_idx, total_steps)

                # --- 核心：应用 Mixup ---
                inputs, targets_a, targets_b, lam = mixup_data(data, emo_labels, alpha=0.3)

                optimizer.zero_grad()
                emo_logits, dom_logits = model(inputs, alpha)

                # 计算混合损失
                loss_emo = mixup_criterion(criterion, emo_logits, targets_a, targets_b, lam)
                loss_dom = criterion(dom_logits, dom_labels)

                # 域对抗损失权重设为 0.5 避免主任务被干扰
                loss = loss_emo + 0.5 * loss_dom

                loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # --- 验证阶段 ---
            model.eval()
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for val_data, val_emo_labels, _, _, _ in val_loader:
                    val_data, val_emo_labels = val_data.to(device), val_emo_labels.to(device)
                    val_emo_logits, _ = model(val_data, alpha=0.0)

                    preds = torch.argmax(val_emo_logits, dim=1)
                    correct_val += (preds == val_emo_labels).sum().item()
                    total_val += val_emo_labels.size(0)

            val_acc = correct_val / total_val * 100 if total_val > 0 else 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print(
                    f"Epoch [{epoch + 1:02d}/{EPOCHS}] | Train Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        print(f"被试 {val_subject} 最终选取最优测试准确率: {best_val_acc:.2f}%")
        fold_results.append(best_val_acc)

    if fold_results:
        print("\n" + "🌟" * 25)
        print(f"LOSO 交叉验证完成！平均跨被试准确率: {np.mean(fold_results):.2f}%")
        print("🌟" * 25)


if __name__ == "__main__":
    train_loso_cv()