import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from .trainer import ModelTrainer


class StrictLOSOCrossValidator:
    """
    严谨的留一被试交叉验证引擎 (Train: N-2, Val: 1, Test: 1)
    绝对物理隔离，防止任何个体级别的信息泄露。
    """

    def __init__(self, dataset, batch_size, epochs, device, save_path):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.save_path = save_path

        # 提取所有不重复的被试 ID 并排序
        self.all_subjects = list(set([sample['subject_id'] for sample in self.dataset.samples]))
        self.all_subjects.sort()

    def run(self, build_components_fn):
        """
        执行交叉验证。
        :param build_components_fn: 一个函数，调用后返回 (model, optimizer, scheduler, criterion)
        """
        fold_results = []
        num_subjects = len(self.all_subjects)

        for fold in range(num_subjects):
            # 1. 严格划分 Train, Val, Test 被试 (无需随机 split)
            test_subject = self.all_subjects[fold]
            val_subject_idx = (fold + 1) % num_subjects
            val_subject = self.all_subjects[val_subject_idx]
            train_subjects = [sub for sub in self.all_subjects if sub not in (test_subject, val_subject)]

            print(f"\n{'=' * 70}")
            print(f"🚀 Fold {fold + 1}/{num_subjects}")
            print(f"🧠 [Train] Subjects ({len(train_subjects)}): {', '.join(train_subjects)}")
            print(f"👁️‍🗨️ [Val]   Subject  (1): {val_subject} (Used for picking best epoch)")
            print(f"🎯 [Test]  Subject  (1): {test_subject} (Absolute Unseen Blind Test)")
            print(f"{'=' * 70}")

            # 2. 直接根据 subject_id 提取数据索引并生成 Dataloader
            train_indices = [i for i, sample in enumerate(self.dataset.samples) if
                             sample['subject_id'] in train_subjects]
            val_indices = [i for i, sample in enumerate(self.dataset.samples) if sample['subject_id'] == val_subject]
            test_indices = [i for i, sample in enumerate(self.dataset.samples) if sample['subject_id'] == test_subject]

            train_loader = DataLoader(Subset(self.dataset, train_indices), batch_size=self.batch_size, shuffle=True,
                                      drop_last=True)
            val_loader = DataLoader(Subset(self.dataset, val_indices), batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(Subset(self.dataset, test_indices), batch_size=self.batch_size, shuffle=False)

            # 3. 核心：通过工厂函数获取【全新初始化】的模型和优化器
            model, optimizer, scheduler, criterion = build_components_fn()

            # 4. 实例化 Trainer 并执行训练与测试
            trainer = ModelTrainer(model, optimizer, scheduler, criterion, self.device, self.save_path)

            print(">>> 阶段一：内部训练与最优参数挑选...")
            trainer.fit(train_loader, val_loader, self.epochs)

            print("\n>>> 阶段二：使用选出的最优参数，对未见被试进行终极盲测...")
            test_acc = trainer.test(test_loader)

            print(f"💡 目标被试 {test_subject} 真实测试准确率: {test_acc:.2f}%\n")
            fold_results.append(test_acc)

        # 5. 打印最终汇总报告
        if fold_results:
            print("🌟" * 25)
            print(f"严谨 LOSO 交叉验证完成！客观平均跨被试准确率: {np.mean(fold_results):.2f}%")
            print("🌟" * 25)

        return fold_results