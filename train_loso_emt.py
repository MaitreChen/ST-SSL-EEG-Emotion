import torch
import torch.nn as nn
import torch.optim as optim
import os

# 导入各类组件
from dataset.data_loader import EEGEmoDataset
from models.emt_wrapper import EndToEndEmT
from utils.loso_cv import StrictLOSOCrossValidator


def main():
    # ================= 1. 超参数配置 =================
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 3e-4
    DATA_ROOT = './data'
    MODEL_SAVE_PATH = './checkpoints/strict_best_model.pth'

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ================= 2. 加载全量数据 =================
    full_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train', crop_len=2500)

    # ================= 3. 定义模型构造器 =================
    def build_components():
        """
        每次调用都会返回一个全新的模型、优化器和调度器。
        这是防止多折交叉验证时“权重污染”的核心机制。
        """
        model = EndToEndEmT(sequence_len=10, num_chan=30, num_class=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, scheduler, criterion

    # ================= 4. 启动交叉验证引擎 =================
    validator = StrictLOSOCrossValidator(
        dataset=full_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=device,
        save_path=MODEL_SAVE_PATH
    )

    # 将模型构造器扔给引擎，自动完成所有折数的跑批
    validator.run(build_components_fn=build_components)


if __name__ == "__main__":
    main()