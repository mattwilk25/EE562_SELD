# model.py — Baseline CRNN for stereo SELD (audio-only, Multi-ACCDOA output).
# Architecture: SELDnet (Adavanne et al., JSTSP 2018) https://arxiv.org/pdf/1807.00129.pdf
# Adapted from the DCASE 2025 Task 3 baseline: https://github.com/sharathadavanne/seld-dcase2025

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CRNNBaseline(nn.Module):
    """Baseline CRNN: conv encoder → BiGRU → MHSA → output head."""

    def __init__(self, params):
        super().__init__()
        C       = params['nb_conv_filters']
        nb_cls  = params['nb_classes']
        dropout = params['dropout']

        # Convolutional encoder
        self.convs = nn.ModuleList()
        in_ch = 2
        for i in range(params['nb_conv_blocks']):
            pool = (params['t_pool_size'][i], params['f_pool_size'][i])
            self.convs.append(AudioConvBlock(in_ch, C, pool, dropout))
            in_ch = C

        f_out = params['nb_mels']
        for f in params['f_pool_size']:
            f_out //= f
        rnn_sz = params['rnn_size']

        self.gru = nn.GRU(C * f_out, rnn_sz, params['nb_rnn_layers'],
                          batch_first=True, dropout=dropout, bidirectional=True)

        self.attn  = nn.ModuleList([
            nn.MultiheadAttention(rnn_sz, params['nb_attn_heads'],
                                  dropout=dropout, batch_first=True)
            for _ in range(params['nb_self_attn_layers'])
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(rnn_sz) for _ in range(params['nb_self_attn_layers'])
        ])

        fnn_in = rnn_sz
        self.fnn = nn.ModuleList()
        for _ in range(params['nb_fnn_layers']):
            self.fnn.append(nn.Linear(fnn_in, params['fnn_size']))
            fnn_in = params['fnn_size']

        self.nb_cls  = nb_cls
        self.head    = nn.Linear(fnn_in, params['max_polyphony'] * 3 * nb_cls)
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, _=None):
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

        x, _ = self.gru(x)
        x = torch.tanh(x)
        half = x.shape[-1] // 2
        x = x[:, :, half:] * x[:, :, :half]

        for attn, ln in zip(self.attn, self.norms):
            h, _ = attn(x, x, x)
            x = ln(x + h)

        for fc in self.fnn:
            x = fc(x)

        B, T, _ = self.head(x).shape
        pred = self.head(x).reshape(B, T, 3, 3, self.nb_cls)
        doa  = self.doa_act(pred[:, :, :, :2, :])
        dist = self.dist_act(pred[:, :, :, 2:, :])
        return torch.cat((doa, dist), dim=3).reshape(B, T, -1)


# Alias used by main.py
SELDModel = CRNNBaseline
