import os
import random
import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EEGEmoDataset(Dataset):
    def __init__(self, data_root, mode='train', crop_len=2500):
        """
        EmoTeam - 脑电情绪识别数据集加载器 (兼容 MATLAB v7.3)
        :param data_root: 数据集的根目录 (例如 './data')
        :param mode: 'train' 或 'test'
        """
        self.mode = mode
        self.samples = []
        self.crop_len = crop_len

        if self.mode == 'train':
            train_dir = os.path.join(data_root, 'train')
            groups = ['HC', 'MDD']

            for group in groups:
                group_dir = os.path.join(train_dir, group)
                if not os.path.exists(group_dir):
                    continue

                for file_name in os.listdir(group_dir):
                    if not file_name.endswith('.mat'):
                        continue

                    file_path = os.path.join(group_dir, file_name)
                    # 提取被试ID，例如 'HC1003'
                    subject_id = file_name.replace('timedata.mat', '')

                    try:
                        data_neu = None
                        data_pos = None

                        try:
                            # 1. 优先尝试 scipy.io 读取 (对于非 v7.3 格式)
                            mat_data = sio.loadmat(file_path)
                            data_neu = mat_data['EEG_data_neu']  # 形状: (30, 50000)
                            data_pos = mat_data['EEG_data_pos']  # 形状: (30, 50000)
                        except NotImplementedError:
                            # 2. 如果报错 "Please use HDF reader"，则使用 h5py 读取
                            with h5py.File(file_path, 'r') as f:
                                # 注意雷区：h5py 读出来的形状是 (50000, 30)
                                # 必须使用 .T 转置回 (30, 50000) 才能保证切片逻辑正确
                                data_neu = np.array(f['EEG_data_neu']).T
                                data_pos = np.array(f['EEG_data_pos']).T

                        if data_neu is None or data_pos is None:
                            continue

                        # 切片：每个类别4段视频，每段 12500 个采样点
                        for i in range(4):
                            start_idx = i * 12500
                            end_idx = (i + 1) * 12500

                            # 提取中性片段 (Label 0)
                            trial_neu = data_neu[:, start_idx:end_idx]
                            self.samples.append({
                                'data': trial_neu,
                                'label': 0,
                                'subject_id': subject_id,
                                'group': group,  # HC 或 MDD
                                'trial_id': i + 1
                            })

                            # 提取积极片段 (Label 1)
                            trial_pos = data_pos[:, start_idx:end_idx]
                            self.samples.append({
                                'data': trial_pos,
                                'label': 1,
                                'subject_id': subject_id,
                                'group': group,
                                'trial_id': i + 1 + 4
                            })
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        elif self.mode == 'test':
            test_dir = os.path.join(data_root, 'test')
            if os.path.exists(test_dir):
                for file_name in os.listdir(test_dir):
                    if not file_name.endswith('.mat'):
                        continue

                    file_path = os.path.join(test_dir, file_name)
                    # 提取测试用户ID，例如 'P_test1'
                    subject_id = file_name.replace('.mat', '')

                    try:
                        test_data = None

                        try:
                            # 优先尝试 scipy.io
                            mat_data = sio.loadmat(file_path)
                            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                            if data_keys:
                                test_data = mat_data[data_keys[0]]  # 形状: (30, 20000)
                        except NotImplementedError:
                            # 回退到 h5py
                            with h5py.File(file_path, 'r') as f:
                                data_keys = list(f.keys())
                                if data_keys:
                                    test_data = np.array(f[data_keys[0]]).T  # 转置回 (30, 20000)

                        if test_data is None:
                            continue

                        # 切片：8段视频，每段 2500 个采样点
                        for i in range(8):
                            start_idx = i * 2500
                            end_idx = (i + 1) * 2500
                            trial_data = test_data[:, start_idx:end_idx]

                            self.samples.append({
                                'data': trial_data,
                                'label': -1,  # 测试集没有真实标签
                                'subject_id': subject_id,
                                'trial_id': i + 1
                            })
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 转为 PyTorch 张量
        data = sample['data'] # 原始形状 (30, Length)

        # --- 最高优先级改进：训练阶段随机裁剪 ---
        if self.mode == 'train':
            max_start = data.shape[1] - self.crop_len  # 12500 - 2500 = 10000
            start_idx = random.randint(0, max_start)
            data = data[:, start_idx: start_idx + self.crop_len]

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # ----------------- 新增：EEG Z-Score 通道级标准化 -----------------
        # 沿着时间维度 (dim=1) 计算每个通道的均值和标准差
        mean = data_tensor.mean(dim=1, keepdim=True)
        std = data_tensor.std(dim=1, keepdim=True) + 1e-6  # 加上 1e-6 防止除以零

        # 对数据进行标准化: (x - mean) / std
        data_tensor = (data_tensor - mean) / std
        # ------------------------------------------------------------------

        label_tensor = torch.tensor(sample['label'], dtype=torch.long)

        return data_tensor, label_tensor, sample['subject_id'], sample.get('group', 'Unknown'), sample['trial_id']


# ----------------- 独立测试运行区 -----------------
if __name__ == "__main__":
    DATA_ROOT = './data'  # 请确保该目录层级与你的当前路径对应

    print("Initializing Training Dataset...")
    train_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='train')
    print(f"Total training trials: {len(train_dataset)}")

    if len(train_dataset) > 0:
        data, label, sub, group, trial = train_dataset[0]
        print(f"Sample Train Data Shape: {data.shape}")
        print(f"Sample Label: {label}, Subject: {sub}, Group: {group}")

    print("\nInitializing Testing Dataset...")
    test_dataset = EEGEmoDataset(data_root=DATA_ROOT, mode='test')
    print(f"Total testing trials: {len(test_dataset)}")

    if len(test_dataset) > 0:
        data, label, sub, group, trial = test_dataset[0]
        print(f"Sample Test Data Shape: {data.shape}")
        print(f"Sample Subject: {sub}, Trial ID: {trial}")