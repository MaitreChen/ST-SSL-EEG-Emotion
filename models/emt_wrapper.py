import torch
import torch.nn as nn
from einops import rearrange

from .emt_core.EmT import EmT


class TGC_FeatureExtractor(nn.Module):
    def __init__(self, seq_len=10, fs=250, channels=30):
        super().__init__()
        self.seq_len = seq_len
        self.fs = fs
        self.channels = channels

    def forward(self, x):
        # x shape: (Batch, 30, 2500)
        B, C, L = x.shape
        chunk_size = L // self.seq_len
        x = x.view(B, C, self.seq_len, chunk_size)
        x = rearrange(x, 'b c s l -> (b c s) l')

        # FFT 提取频域能量
        fft_vals = torch.fft.rfft(x, dim=-1)
        psd = torch.abs(fft_vals) ** 2
        freqs = torch.fft.rfftfreq(chunk_size, 1 / self.fs)

        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
        features = []
        for f_min, f_max in bands.values():
            idx = torch.where((freqs >= f_min) & (freqs <= f_max))[0]
            if len(idx) > 0:
                band_power = psd[:, idx].mean(dim=-1, keepdim=True)
            else:
                band_power = torch.zeros((psd.shape[0], 1), device=x.device)
            features.append(band_power)

        features = torch.cat(features, dim=-1)  # (b*c*s, 5)
        features = rearrange(features, '(b c s) f -> b s c f', b=B, c=C, s=self.seq_len)
        features = torch.log(features + 1e-7)
        return features


class EndToEndEmT(nn.Module):
    def __init__(self, sequence_len=10, num_chan=30, num_class=2):
        super().__init__()
        self.feature_extractor = TGC_FeatureExtractor(seq_len=sequence_len, channels=num_chan)

        self.emt_model = EmT(
            layers_graph=[1, 2],
            layers_transformer=4,  # 防止显存溢出设为4
            num_adj=2,
            num_chan=num_chan,
            num_feature=5,
            hidden_graph=32,
            K=3,
            num_head=8,
            dim_head=32,
            dropout=0.3,
            num_class=num_class,
            graph2token='Linear',
            encoder_type='Cheby',
            alpha=0.25
        )

    def forward(self, x):
        graph_features = self.feature_extractor(x)
        out = self.emt_model(graph_features)
        return out
