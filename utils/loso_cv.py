import torch
from torch.utils.data import DataLoader
import numpy as np
from .trainer import ModelTrainer
from dataset.data_loader import EEGEmoDataset  # 导入更新后的Dataset


class StrictLOSOCrossValidator:
    """
    竞赛级留一被试验证引擎 (Train: N-2, Val: 1, Test: 1)
    完美匹配 10秒 盲测环境
    """

    def __init__(self, raw_samples, batch_size, epochs, device, save_path):
        self.raw_samples = raw_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.save_path = save_path

        # 提取所有不重复的被试 ID 并排序
        self.all_subjects = list(set([s['subject_id'] for s in self.raw_samples]))
        self.all_subjects.sort()

    def run(self, build_components_fn):
        fold_results = []
        num_subjects = len(self.all_subjects)

        for fold in range(num_subjects):
            test_subject = self.all_subjects[fold]
            val_subject_idx = (fold + 1) % num_subjects
            val_subject = self.all_subjects[val_subject_idx]
            train_subjects = [sub for sub in self.all_subjects if sub not in (test_subject, val_subject)]

            print(f"\n{'=' * 75}")
            print(f"🚀 Fold {fold + 1}/{num_subjects}")
            print(f"🧠 [Train] Subjects: {len(train_subjects)} | Mode: Sliding Window 3000 + Random Crop 2500")
            print(f"👁️‍🗨️ [Val]   Subject : {val_subject} | Mode: Strict 10s Split (For Model Selection)")
            print(f"🎯 [Test]  Subject : {test_subject} | Mode: Strict 10s Split (Absolute Blind Test)")
            print(f"{'=' * 75}")

            # 1. 过滤原始内存数据池
            train_raw = [s for s in self.raw_samples if s['subject_id'] in train_subjects]
            val_raw = [s for s in self.raw_samples if s['subject_id'] == val_subject]
            test_raw = [s for s in self.raw_samples if s['subject_id'] == test_subject]

            # 2. 赋予 Dataset 不同的 mode 策略
            train_dataset = EEGEmoDataset(train_raw, mode='train', crop_len=2500)
            val_dataset = EEGEmoDataset(val_raw, mode='val', crop_len=2500)
            test_dataset = EEGEmoDataset(test_raw, mode='test', crop_len=2500)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # 3. 核心工厂函数：获取全新初始化的模型
            model, optimizer, scheduler, criterion = build_components_fn()
            trainer = ModelTrainer(model, optimizer, scheduler, criterion, self.device, self.save_path)

            print(f">>> 阶段一：训练与验证 (当前训练批次量: {len(train_loader)} batches)...")
            trainer.fit(train_loader, val_loader, self.epochs)

            print("\n>>> 阶段二：加载最优参数，进行终极盲测...")
            test_acc = trainer.test(test_loader)
            print(f"💡 目标被试 {test_subject} 真实测试准确率: {test_acc:.2f}%\n")

            fold_results.append(test_acc)

        if fold_results:
            print("🌟" * 25)
            print(f"严谨 LOSO 交叉验证完成！完全模拟 10s 截断比赛条件。")
            print(f"客观平均跨被试准确率: {np.mean(fold_results):.2f}%")
            print("🌟" * 25)

        return fold_results