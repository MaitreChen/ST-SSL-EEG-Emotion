import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from data_loader import EEGEmoDataset
from vit_backbone import EEGTemporalViT
from masked_pretrain import EEGMaskedAutoencoder


def run_pretraining():
    # 超参数设置
    PRETRAIN_EPOCHS = 50
    BATCH_SIZE = 32
    LR = 3e-4
    DATA_ROOT = './data'
    SAVE_DIR = './weights'

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device for Pre-training: {device}")

    # 1. 分别加载 Train 和 Test 数据集 (避免同一批次内长度不一)
    print("正在分别加载 Train 和 Test 数据集用于联合预训练...")
    train_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train')
    test_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='test')

    # 分别创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"预训练准备就绪：Train 样本 {len(train_dataset)} 个，Test 样本 {len(test_dataset)} 个")

    # 2. 初始化网络模型
    backbone = EEGTemporalViT(in_channels=30, patch_size=50, embed_dim=256)
    mae_model = EEGMaskedAutoencoder(backbone, mask_ratio=0.4).to(device)
    optimizer = optim.AdamW(mae_model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    # 3. 开始预训练循环
    print("\n" + "=" * 50)
    print("🔥 开始时空联合掩码自监督预训练 (Masked Pre-training)")
    print("=" * 50)

    for epoch in range(PRETRAIN_EPOCHS):
        mae_model.train()
        total_loss = 0.0
        total_batches = 0

        # --- 遍历训练集批次 (长度为 12500) ---
        for batch_idx, (data, _, _, _, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            pred, target, mask = mae_model(data)

            loss_matrix = criterion(pred, target)
            loss = (loss_matrix * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]
            # loss = (pred - target) ** 2
            # loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        # --- 遍历测试集批次 (长度为 2500) ---
        # 我们的基座支持变长输入，这里直接喂入更短的数据毫无压力！
        for batch_idx, (data, _, _, _, _) in enumerate(test_loader):
            data = data.to(device)
            optimizer.zero_grad()

            pred, target, mask = mae_model(data)

            loss_matrix = criterion(pred, target)
            loss = (loss_matrix * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]
            # loss = (pred - target) ** 2
            # loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum() / target.shape[-1]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Epoch [{epoch + 1}/{PRETRAIN_EPOCHS}] | Reconstruction Loss: {avg_loss:.4f}")

    # 4. 保存预训练好的 Backbone 权重
    save_path = os.path.join(SAVE_DIR, 'pretrained_backbone.pth')
    torch.save(mae_model.backbone.state_dict(), save_path)
    print("\n✅ 预训练完成！Backbone 权重已保存至:", save_path)


if __name__ == "__main__":
    run_pretraining()