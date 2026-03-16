import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer


class MCSA(nn.Module):
    """Multi-Scale Channel Attention Module.

    Computes channel attention using both a global branch (GAP + pointwise conv)
    and a local branch (pointwise conv), then fuses them via sigmoid gating.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def compute_attention(self, x):
        g = self.global_branch(x)
        l = self.local_branch(x)
        return self.sigmoid(g + l)

    def forward(self, x):
        return x * self.compute_attention(x)


class MCSANetBlock(nn.Module):
    """Single ResNet-style block with MCSA and frequency-only pooling.

    Structure: Conv-BN-ReLU -> Conv-BN -> + residual -> ReLU -> MCSA -> F-Pool -> Dropout
    """

    def __init__(self, in_channels, out_channels, f_pool_size=2,
                 dropout=0.05, mcsa_reduction=4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.mcsa = MCSA(out_channels, reduction=mcsa_reduction)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.f_pool = nn.MaxPool2d((1, f_pool_size)) if f_pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        out = self.mcsa(out)
        out = self.f_pool(out)
        out = self.dropout(out)
        return out


class MCSANet(nn.Module):
    """MCSAnet backbone: stacked ResNet-MCSA blocks with frequency-only pooling.

    Processes single-channel spectrogram input while preserving full temporal
    resolution. Only frequency dimension is reduced through pooling.
    """

    def __init__(self, in_channels=1, channels=None, f_pool_sizes=None,
                 dropout=0.05, mcsa_reduction=4):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 256]
        if f_pool_sizes is None:
            f_pool_sizes = [4, 4, 2, 1]

        self.blocks = nn.ModuleList()
        for i, (ch, fp) in enumerate(zip(channels, f_pool_sizes)):
            in_ch = in_channels if i == 0 else channels[i - 1]
            self.blocks.append(
                MCSANetBlock(in_ch, ch, f_pool_size=fp,
                             dropout=dropout, mcsa_reduction=mcsa_reduction)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AFF(nn.Module):
    """Attentional Feature Fusion.

    Adaptively fuses left and right channel features using MCSA-generated weights.
    F_fused = w * F_left + (1 - w) * F_right
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.mcsa = MCSA(channels, reduction=reduction)

    def forward(self, x_left, x_right):
        combined = x_left + x_right
        w = self.mcsa.compute_attention(combined)
        return w * x_left + (1.0 - w) * x_right


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal temporal positional encoding."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StereoRCnet(nn.Module):
    """Stereo ResNet-Conformer network for Sound Event Localization and Detection.

    Architecture:
        Left/Right stereo -> MCSAnet (shared weights) -> AFF fusion ->
        Embedding -> T-Encoding -> Conformer -> T-Pooling -> FC ->
        Single ACCDOA output (x, y, dist per class)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nb_classes = cfg['nb_classes']

        # Shared-weight MCSAnet for stereo channels
        self.mcsanet = MCSANet(
            in_channels=1,
            channels=cfg['mcsanet_channels'],
            f_pool_sizes=cfg['f_pool_sizes'],
            dropout=cfg['dropout'],
            mcsa_reduction=cfg['mcsa_reduction'],
        )

        # Compute flattened feature dimension after MCSAnet
        final_ch = cfg['mcsanet_channels'][-1]
        final_freq = cfg['nb_mels']
        for fp in cfg['f_pool_sizes']:
            final_freq = final_freq // fp
        self.feat_dim = final_ch * final_freq

        # Attentional Feature Fusion for stereo channels
        self.aff = AFF(final_ch, reduction=cfg['mcsa_reduction'])

        # Embedding projection
        conformer_dim = cfg['conformer_dim']
        self.embedding = nn.Linear(self.feat_dim, conformer_dim)

        # Temporal positional encoding (fixed sinusoidal)
        self.pos_encoding = PositionalEncoding(conformer_dim, max_len=cfg['nb_frames'] + 10)

        # Conformer encoder
        self.conformer = Conformer(
            input_dim=conformer_dim,
            num_heads=cfg['conformer_heads'],
            ffn_dim=cfg['conformer_ffn_dim'],
            num_layers=cfg['conformer_layers'],
            depthwise_conv_kernel_size=cfg['conformer_depthwise_kernel'],
            dropout=cfg['conformer_dropout'],
        )

        # Temporal pooling: 251 frames -> 50 label frames
        self.t_pool = nn.AdaptiveAvgPool1d(cfg['label_sequence_length'])

        # Output projection: single ACCDOA = (x, y, dist) per class
        output_dim = 3 * nb_classes
        self.fc_out = nn.Linear(conformer_dim, output_dim)

        # Activation functions
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, 2, T, F) stereo log-mel spectrogram, T=251, F=64
        Returns:
            pred: (B, 50, 3*nb_classes) single ACCDOA output
        """
        B = x.size(0)
        nb_classes = self.cfg['nb_classes']

        # Split stereo channels
        x_left = x[:, 0:1, :, :]   # (B, 1, T, F)
        x_right = x[:, 1:2, :, :]  # (B, 1, T, F)

        # MCSAnet feature extraction with shared weights
        feat_left = self.mcsanet(x_left)    # (B, C, T, F')
        feat_right = self.mcsanet(x_right)  # (B, C, T, F')

        # Attentional Feature Fusion
        fused = self.aff(feat_left, feat_right)  # (B, C, T, F')

        # Reshape to temporal sequence: (B, C, T, F') -> (B, T, C*F')
        fused = fused.permute(0, 2, 1, 3).contiguous()
        fused = fused.view(B, fused.size(1), -1)

        # Embedding + temporal positional encoding
        h = self.embedding(fused)       # (B, T, D)
        h = self.pos_encoding(h)        # (B, T, D)

        # Conformer encoder
        T = h.size(1)
        lengths = torch.full((B,), T, dtype=torch.long, device=h.device)
        h, _ = self.conformer(h, lengths)  # (B, T, D)

        # Temporal pooling: (B, T, D) -> (B, 50, D)
        h = h.transpose(1, 2)              # (B, D, T)
        h = self.t_pool(h)                 # (B, D, 50)
        h = h.transpose(1, 2).contiguous() # (B, 50, D)

        # Output projection
        pred = self.fc_out(h)  # (B, 50, 3*nb_classes)

        # Apply activation functions per component
        pred = pred.view(B, -1, 3, nb_classes)
        doa = self.doa_act(pred[:, :, :2, :])     # Tanh for DOA (x, y)
        dist = self.dist_act(pred[:, :, 2:3, :])  # ReLU for distance
        pred = torch.cat([doa, dist], dim=2)
        pred = pred.view(B, -1, 3 * nb_classes)

        return pred


if __name__ == '__main__':
    from config import config

    model = StereoRCnet(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n_params:,}')

    dummy = torch.randn(4, 2, 251, 64)
    out = model(dummy)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {out.shape}')
