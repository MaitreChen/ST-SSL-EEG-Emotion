import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入我们之前写好的模块
from data_loader import EEGEmoDataset
from vit_backbone import EEGTemporalViT
from domain_adaptation import DomainDiscriminator, get_alpha


# ----------------- 1. 情绪分类头 -----------------
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=128):
        """
        EmoTeam - 二分类情绪判别头 (积极 vs 中性)
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 输出2个类别：0(中性), 1(积极)
        )

    def forward(self, x):
        return self.classifier(x)


# ----------------- 2. 统一框架封装 -----------------
class EmoTeamFramework(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 共享特征提取基座
        self.backbone = EEGTemporalViT(in_channels=30, patch_size=50, embed_dim=256)
        # 2. 任务分类头 (Emotion)
        self.classifier = EmotionClassifier(input_dim=256)
        # 3. 对抗域判别器 (Domain: HC vs MDD)
        self.domain_discriminator = DomainDiscriminator(input_dim=256)

    def forward(self, x, alpha):
        # 提取全局特征
        cls_features, _ = self.backbone(x)
        # 情绪分类预测
        emotion_logits = self.classifier(cls_features)
        # 域分类预测 (经过梯度反转)
        domain_logits = self.domain_discriminator(cls_features, alpha)

        return emotion_logits, domain_logits


# ----------------- 3. 训练循环 (Training Loop) -----------------
def train_baseline():
    # 超参数设置
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 1e-4
    DATA_ROOT = './data'

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 加载数据
    print("加载训练数据集...")
    train_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 实例化模型与优化器
    model = EmoTeamFramework().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 损失函数
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    total_steps = EPOCHS * len(train_loader)
    current_step = 0

    print("开始训练 Baseline 模型...")
    for epoch in range(EPOCHS):
        model.train()
        total_emo_loss = 0.0
        total_dom_loss = 0.0
        correct_emo = 0
        total_samples = 0

        for batch_idx, (data, emo_labels, _, group_labels, _) in enumerate(train_loader):
            data, emo_labels = data.to(device), emo_labels.to(device)

            # 将 HC 和 MDD 转换为域标签: HC -> 0, MDD -> 1
            dom_labels = torch.tensor([0 if g == 'HC' else 1 for g in group_labels]).to(device)

            # 动态计算当前的 alpha 值
            current_step += 1
            alpha = get_alpha(current_step, total_steps)

            # 前向传播
            optimizer.zero_grad()
            emotion_logits, domain_logits = model(data, alpha)

            # 计算损失
            loss_emo = criterion_emotion(emotion_logits, emo_labels)
            loss_dom = criterion_domain(domain_logits, dom_labels)

            # 总损失：分类损失 + 域对抗损失
            loss = loss_emo + loss_dom

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 统计指标
            total_emo_loss += loss_emo.item()
            total_dom_loss += loss_dom.item()
            preds = torch.argmax(emotion_logits, dim=1)
            correct_emo += (preds == emo_labels).sum().item()
            total_samples += emo_labels.size(0)

        # 打印 Epoch 级指标
        epoch_acc = correct_emo / total_samples * 100
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Alpha: {alpha:.3f} | "
              f"Emo Loss: {total_emo_loss / len(train_loader):.4f} | "
              f"Dom Loss: {total_dom_loss / len(train_loader):.4f} | "
              f"Emo Train Acc: {epoch_acc:.2f}%")


if __name__ == "__main__":
    train_baseline()