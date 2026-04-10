import torch
import os
from .augment import mixup_data, mixup_criterion


class ModelTrainer:
    """
    通用深度学习模型训练器
    负责处理 Epoch 循环、验证集指标监控、模型早停与保存
    """

    def __init__(self, model, optimizer, scheduler, criterion, device, save_path):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

    def train_epoch(self, dataloader):
        self.model.train()
        train_loss = 0.0

        for data, emo_labels, _, _, _ in dataloader:
            data, emo_labels = data.to(self.device), emo_labels.to(self.device)

            # 统一应用 Mixup
            inputs, targets_a, targets_b, lam = mixup_data(data, emo_labels, alpha=0.3)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            train_loss += loss.item()

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

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)

            # 苛刻的模型挑选条件
            if val_acc >= best_val_acc and epoch >= int(epochs * 0.2):
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Internal Val Acc: {val_acc:.2f}%")

        return best_val_acc

    def test(self, test_loader):
        """阶段二：加载最优权重并盲测"""
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
        else:
            print("警告: 采用最后一次训练轮次的权重进行测试。")

        test_acc = self.evaluate(test_loader)
        return test_acc