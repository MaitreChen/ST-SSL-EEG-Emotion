import torch
import os
from tqdm import tqdm
from .augment import mixup_data, mixup_criterion


class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, save_path):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        train_loss = 0.0

        # 包装 DataLoader，设置 dynamic_ncols 自动适应终端宽度，leave=False 让跑完的进度条消失保持控制台整洁
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}]", leave=False, dynamic_ncols=True)

        for data, emo_labels, _, _, _ in pbar:
            data, emo_labels = data.to(self.device), emo_labels.to(self.device)
            inputs, targets_a, targets_b, lam = mixup_data(data, emo_labels, alpha=0.3)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if self.scheduler:
            self.scheduler.step()

        return train_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for data, labels, _, _, _ in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits = self.model(data)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return (correct / total * 100) if total > 0 else 0

    def fit(self, train_loader, val_loader, epochs):
        """阶段一：包含内部验证和最优挑选的完整训练流"""
        best_val_acc = 0.0

        # 将 epoch 从 1 开始计数，方便 tqdm 显示
        for epoch in range(1, epochs + 1):
            # 传入当前的 epoch 和总 epochs 以便 tqdm 显示
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            val_acc = self.evaluate(val_loader)

            # 苛刻的模型挑选条件
            if val_acc > best_val_acc and epoch >= int(epochs * 0.2):
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)

                # 每 10 个 Epoch 打印一次静态日志，不会干扰 tqdm
            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Internal Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        return best_val_acc

    def test(self, test_loader):
        """阶段二：加载最优权重并盲测"""
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
            print("  --> Successfully loaded the best model weights for testing.")
        else:
            print("  --> WARNING: Target weight not found. Using the latest weights.")

        test_acc = self.evaluate(test_loader)
        return test_acc