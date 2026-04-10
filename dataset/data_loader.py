import os
import random
import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_all_competition_train_data(data_root):
    """
    一次性将所有训练集数据（50秒/12500点）加载到内存字典中，极大加速后续的交叉验证。
    返回: raw_samples 列表，每个元素是一段完整的 50秒 trial
    """
    train_dir = os.path.join(data_root, 'train')
    groups = ['HC', 'MDD']
    raw_samples = []

    for group in groups:
        group_dir = os.path.join(train_dir, group)
        if not os.path.exists(group_dir):
            continue

        for file_name in os.listdir(group_dir):
            if not file_name.endswith('.mat'):
                continue

            file_path = os.path.join(group_dir, file_name)
            subject_id = file_name.replace('timedata.mat', '')

            try:
                data_neu, data_pos = None, None
                try:
                    mat_data = sio.loadmat(file_path)
                    data_neu = mat_data['EEG_data_neu']  # (30, 50000)
                    data_pos = mat_data['EEG_data_pos']  # (30, 50000)
                except NotImplementedError:
                    with h5py.File(file_path, 'r') as f:
                        data_neu = np.array(f['EEG_data_neu']).T
                        data_pos = np.array(f['EEG_data_pos']).T

                if data_neu is None or data_pos is None:
                    continue

                # 提取完整的 50 秒片段 (12500点)
                for i in range(4):
                    start_idx = i * 12500
                    end_idx = (i + 1) * 12500

                    # 中性情绪 (Label 0)
                    raw_samples.append({
                        'data': data_neu[:, start_idx:end_idx],
                        'label': 0, 'subject_id': subject_id, 'group': group, 'trial_id': i + 1
                    })
                    # 积极情绪 (Label 1)
                    raw_samples.append({
                        'data': data_pos[:, start_idx:end_idx],
                        'label': 1, 'subject_id': subject_id, 'group': group, 'trial_id': i + 5
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return raw_samples


class EEGEmoDataset(Dataset):
    def __init__(self, raw_samples, mode='train', crop_len=2500):
        """
        针对竞赛优化的 Dataset
        :param raw_samples: 由 load_all_competition_train_data 提供的数据池
        :param mode: 'train' (滑窗扩充+动态裁剪) 或 'val'/'test' (严格10秒切片)
        """
        self.mode = mode
        self.crop_len = crop_len
        self.samples = []

        for sample in raw_samples:
            raw_data = sample['data']  # 原始维度 (30, 12500)

            if self.mode == 'train':
                # 【核心增强】：滑动窗口 (Sliding Window)
                # 设定稍大一点的窗口(3000)用于后续随机裁剪，步长1000
                win_size = 3000
                stride = 1000
                for start in range(0, raw_data.shape[1] - win_size + 1, stride):
                    self.samples.append({
                        'data': raw_data[:, start: start + win_size],
                        'label': sample['label'],
                        'subject_id': sample['subject_id'],
                        'group': sample['group']
                    })

            elif self.mode in ['val', 'test']:
                # 【严格对齐】：将50秒数据严格且不重叠地切分成5段10秒(2500点)片段
                # 完美模拟比赛官方测试集的输入格式
                win_size = 2500
                stride = 2500
                for start in range(0, raw_data.shape[1] - win_size + 1, stride):
                    self.samples.append({
                        'data': raw_data[:, start: start + win_size],
                        'label': sample['label'],
                        'subject_id': sample['subject_id'],
                        'group': sample['group']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample['data']

        # 【核心增强】：动态随机裁剪 (Dynamic Random Crop)
        if self.mode == 'train':
            max_start = data.shape[1] - self.crop_len  # 3000 - 2500 = 500
            start_idx = random.randint(0, max_start)
            data = data[:, start_idx: start_idx + self.crop_len]

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 独立的通道级 Z-Score 标准化 (抑制个体基线漂移)
        mean = data_tensor.mean(dim=1, keepdim=True)
        std = data_tensor.std(dim=1, keepdim=True) + 1e-6
        data_tensor = (data_tensor - mean) / std

        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        return data_tensor, label_tensor, sample['subject_id'], sample.get('group', 'Unknown'), sample.get('trial_id',
                                                                                                           -1)