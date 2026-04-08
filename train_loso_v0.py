import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# 导入我们之前写好的模块
from data_loader import EEGEmoDataset
from train import EmoTeamFramework  # 直接复用刚才写好的统一框架
from domain_adaptation import get_alpha


def train_loso_cv():
    # 超参数设置
    EPOCHS = 50  # 实际比赛建议调高到 30-50
    BATCH_SIZE = 32
    LR = 1e-4
    DATA_ROOT = './data'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 加载完整训练数据集
    print("加载完整训练数据集...")
    full_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train')

    # 2. 提取所有独立的被试 ID
    all_subjects = list(set([sample['subject_id'] for sample in full_dataset.samples]))
    all_subjects.sort()  # 排序保证每次运行的 fold 顺序一致
    num_subjects = len(all_subjects)
    print(f"共发现 {num_subjects} 名独立被试，准备启动 Leave-One-Subject-Out (LOSO) 交叉验证...")

    fold_results = []  # 记录每次 Fold 的验证准确率

    # 3. 开始 LOSO 循环
    # 注意：为了方便你快速跑通测试，这里加了个小限制，只跑前 3 个被试。
    # 比赛最终跑数据时，请把 `[:3]` 去掉，跑完整的 num_subjects 次循环。
    for fold, val_subject in enumerate(all_subjects[:5]):
        print("\n" + "=" * 50)
        print(f"🚀 Fold {fold + 1}/{num_subjects} | 验证被试 (Unseen): {val_subject}")
        print("=" * 50)

        # 划分 Train 和 Val 的索引
        train_indices = [i for i, sample in enumerate(full_dataset.samples) if sample['subject_id'] != val_subject]
        val_indices = [i for i, sample in enumerate(full_dataset.samples) if sample['subject_id'] == val_subject]

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # 每个 Fold 必须重新初始化模型和优化器，防止特征泄露
        model = EmoTeamFramework().to(device)

        # ---------------- 新增：加载预训练权重 ----------------
        # pretrained_path = './weights/pretrained_backbone.pth'
        # import os  # 确保文件开头导入了 os
        # if os.path.exists(pretrained_path):
        #     print(f"🔄 正在加载预训练 Backbone 权重...")
        #     # 因为框架中基座的变量名是 backbone，所以直接对应加载
        #     model.backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
        #     print("✅ 预训练权重加载成功！")
        # else:
        #     print("⚠️ 未找到预训练权重，将随机初始化从头训练。")
        # ------------------------------------------------------

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion_emotion = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()

        total_steps = EPOCHS * len(train_loader)

        # 训练 Epoch 循环
        for epoch in range(EPOCHS):
            model.train()
            current_step = epoch * len(train_loader)

            for batch_idx, (data, emo_labels, _, group_labels, _) in enumerate(train_loader):
                data, emo_labels = data.to(device), emo_labels.to(device)
                dom_labels = torch.tensor([0 if g == 'HC' else 1 for g in group_labels]).to(device)

                current_step += 1
                alpha = get_alpha(current_step, total_steps)

                optimizer.zero_grad()
                emotion_logits, domain_logits = model(data, alpha)

                loss_emo = criterion_emotion(emotion_logits, emo_labels)
                loss_dom = criterion_domain(domain_logits, dom_labels)
                loss = loss_emo + loss_dom

                loss.backward()
                optimizer.step()

            # 验证阶段 (评估这个“没见过”的被试)
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for val_data, val_emo_labels, _, _, _ in val_loader:
                    val_data, val_emo_labels = val_data.to(device), val_emo_labels.to(device)
                    # 验证时不需要计算对抗域损失，所以 alpha 随便传个 0
                    val_emo_logits, _ = model(val_data, alpha=0.0)
                    preds = torch.argmax(val_emo_logits, dim=1)
                    correct_val += (preds == val_emo_labels).sum().item()
                    total_val += val_emo_labels.size(0)

            val_acc = correct_val / total_val * 100 if total_val > 0 else 0

            # 为了清爽，只打印每个被试最后一个 Epoch 的验证结果
            if epoch == EPOCHS - 1:
                print(f"被试 {val_subject} 的最终测试准确率: {val_acc:.2f}%")
                fold_results.append(val_acc)

    # 打印 LOSO 总体性能
    if fold_results:
        avg_acc = np.mean(fold_results)
        print("\n" + "🌟" * 25)
        print(f"LOSO 交叉验证完成！平均跨被试准确率: {avg_acc:.2f}%")
        print("🌟" * 25)


if __name__ == "__main__":
    train_loso_cv()
