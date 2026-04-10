import torch
import torch.nn as nn
import torch.optim as optim
import os

from dataset.data_loader import load_all_competition_train_data
from models.emt_wrapper import EndToEndEmT
from utils.loso_cv import StrictLOSOCrossValidator


def main():
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 3e-4
    DATA_ROOT = './data'
    MODEL_SAVE_PATH = './checkpoints/strict_best_model.pth'

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ================= 1. 预加载全部数据 (告别 IO 瓶颈) =================
    print("正在将所有训练被试的 50s 脑电数据载入内存...")
    raw_samples = load_all_competition_train_data(data_root=DATA_ROOT)
    print(f"成功加载 {len(raw_samples)} 个完整的 50s Trial。")

    # ================= 2. 定义每次重置的组件构造器 =================
    def build_components():
        model = EndToEndEmT(sequence_len=10, num_chan=30, num_class=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    # ================= 3. 启动竞赛级验证引擎 =================
    validator = StrictLOSOCrossValidator(
        raw_samples=raw_samples,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=device,
        save_path=MODEL_SAVE_PATH
    )

    validator.run(build_components_fn=build_components)


if __name__ == "__main__":
    main()