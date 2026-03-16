"""
ResNet-Conformer SELD model 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Residual Block
# --------------------------------------------------------------------------- #

class ResidualBlock(nn.Module):
    """
    Two-layer residual block: Conv2d → BN → ReLU → Conv2d → BN → (+skip) → ReLU.
    A projection shortcut is added automatically when channel count or spatial
    size changes.
    """
    def __init__(self, in_ch, out_ch, pool_size=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.pool  = nn.MaxPool2d(pool_size) if pool_size != (1, 1) else nn.Identity()

        # Projection shortcut when in_ch != out_ch
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return self.pool(x)


# --------------------------------------------------------------------------- #
# Conformer Block
# --------------------------------------------------------------------------- #

class ConformerBlock(nn.Module):
    """
    Conformer block (Gulati et al. 2020):
        FF (half-step) → MHSA → Conv module → FF (half-step) → LayerNorm

    The depthwise conv captures local temporal patterns while MHSA handles
    long-range dependencies — both lacking individually in BiGRU + MHSA.
    """
    def __init__(self, d_model, n_heads, ff_dim, conv_kernel=31, dropout=0.1):
        super().__init__()

        # Feed-forward 1 (half-step)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mhsa_drop = nn.Dropout(dropout)

        # Convolution module (GLU gating + depthwise conv)
        self.conv_norm  = nn.LayerNorm(d_model)
        self.pw1        = nn.Conv1d(d_model, 2 * d_model, 1)     # pointwise expand (GLU)
        self.dw         = nn.Conv1d(d_model, d_model, conv_kernel,
                                    padding=conv_kernel // 2, groups=d_model)
        self.conv_bn    = nn.BatchNorm1d(d_model)
        self.pw2        = nn.Conv1d(d_model, d_model, 1)          # pointwise project
        self.conv_drop  = nn.Dropout(dropout)

        # Feed-forward 2 (half-step)
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # FF1 half-step
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # MHSA
        h = self.mhsa_norm(x)
        h, _ = self.mhsa(h, h, h)
        x = x + self.mhsa_drop(h)

        # Conv module
        h = self.conv_norm(x).transpose(1, 2)   # B × d × T
        h = self.pw1(h)
        h, gate = h.chunk(2, dim=1)
        h = h * torch.sigmoid(gate)             # GLU
        h = F.silu(self.conv_bn(self.dw(h)))
        h = self.conv_drop(self.pw2(h)).transpose(1, 2)  # B × T × d
        x = x + h

        # FF2 half-step
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class SELDModelImproved(nn.Module):
    """
    ResNet-Conformer SELD model.

    Encoder:   stem conv → 3 × ResidualBlock stages (same pooling schedule as
               baseline to produce T=50 output frames).
    Temporal:  Linear projection → N × ConformerBlock.
    Output:    Linear → same multiACCDOA / singleACCDOA shapes as SELDModel.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        d_model     = params.get('rc_d_model',      256)
        n_heads     = params.get('rc_n_heads',       8)
        ff_dim      = params.get('rc_ff_dim',        512)
        n_conformer = params.get('rc_nb_conformer',  4)
        conv_kernel = params.get('rc_conv_kernel',   31)
        dropout     = params['dropout']

        # ------------------------------------------------------------------- #
        # ResNet encoder
        # Input: B × 2 × 251 × 64
        # ------------------------------------------------------------------- #
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Stage 1: MaxPool(5,4)  →  B × 64  × 50 × 16
        self.stage1 = ResidualBlock(64,  64,  pool_size=(params['t_pool_size'][0], params['f_pool_size'][0]))
        # Stage 2: MaxPool(1,4)  →  B × 128 × 50 × 4
        self.stage2 = ResidualBlock(64,  128, pool_size=(params['t_pool_size'][1], params['f_pool_size'][1]))
        # Stage 3: MaxPool(1,2)  →  B × 256 × 50 × 2
        self.stage3 = ResidualBlock(128, 256, pool_size=(params['t_pool_size'][2], params['f_pool_size'][2]))

        # After flattening frequency dim: 256 × 2 = 512
        cnn_out_dim = 256 * (params['nb_mels'] // (params['f_pool_size'][0] * params['f_pool_size'][1] * params['f_pool_size'][2]))
        self.proj = nn.Linear(cnn_out_dim, d_model)

        # ------------------------------------------------------------------- #
        # Conformer temporal model
        # ------------------------------------------------------------------- #
        self.conformer = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_dim, conv_kernel, dropout)
            for _ in range(n_conformer)
        ])

        # Output head: 3 tracks × (x, y, dist) × nb_classes = 117 values per frame
        out_dim = params['max_polyphony'] * 3 * params['nb_classes']
        self.output_head = nn.Linear(d_model, out_dim)
        self.doa_act     = nn.Tanh()
        self.dist_act    = nn.ReLU()

    def forward(self, x):
        # ResNet encoder
        x = self.stem(x)      # B × 64  × T × F
        x = self.stage1(x)    # B × 64  × 50 × 16
        x = self.stage2(x)    # B × 128 × 50 × 4
        x = self.stage3(x)    # B × 256 × 50 × 2

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x = self.proj(x)      # B × 50 × d_model

        for block in self.conformer:
            x = block(x)

        nb_cls = self.params['nb_classes']
        pred = self.output_head(x).reshape(B, T, 3, 3, nb_cls)
        doa  = self.doa_act(pred[:, :, :, :2, :])
        dist = self.dist_act(pred[:, :, :, 2:, :])
        return torch.cat((doa, dist), dim=3).reshape(B, T, -1)


if __name__ == '__main__':
    params = {
        'nb_mels': 64, 'nb_classes': 13, 'max_polyphony': 3, 'dropout': 0.05,
        'f_pool_size': [4, 4, 2], 't_pool_size': [5, 1, 1],
    }
    model = SELDModelImproved(params)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    out = model(torch.rand(4, 2, 251, 64))
    print(f'Output shape: {out.shape}')   # expect (4, 50, 117)
