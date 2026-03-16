import torch
import torch.nn as nn


class ADPITLoss(nn.Module):
    """
    Auxiliary Duplicating Permutation Invariant Training (ADPIT) loss.

    Generates all 13 valid track permutations of the ground-truth labels,
    picks the one with the lowest MSE, and backpropagates only through that
    assignment.

    Target: (B, T, 6, 4, C)  — 6 dummy tracks, 4 components (act,x,y,dist), C classes
    Output: (B, T, 117)       — 3 tracks × (x,y,dist) × 13 classes
    """

    def __init__(self, params=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    @staticmethod
    def _cat(a, b, c):
        return torch.cat((a, b, c), dim=2)

    def _permutations(self, A0, B0, B1, C0, C1, C2):
        cat = self._cat
        pad_A = cat(B0, B0, B1) + cat(C0, C1, C2)
        pad_B = cat(A0, A0, A0) + cat(C0, C1, C2)
        pad_C = cat(A0, A0, A0) + cat(B0, B0, B1)
        return [
            cat(A0, A0, A0) + pad_A,
            cat(B0, B0, B1) + pad_B, cat(B0, B1, B0) + pad_B,
            cat(B0, B1, B1) + pad_B, cat(B1, B0, B0) + pad_B,
            cat(B1, B0, B1) + pad_B, cat(B1, B1, B0) + pad_B,
            cat(C0, C1, C2) + pad_C, cat(C0, C2, C1) + pad_C,
            cat(C1, C0, C2) + pad_C, cat(C1, C2, C0) + pad_C,
            cat(C2, C0, C1) + pad_C, cat(C2, C1, C0) + pad_C,
        ]

    def forward(self, output, target):
        B, T   = target.shape[:2]
        n_elem = target.shape[3] - 1  # (x,y,dist) — drop activity
        n_cls  = target.shape[4]

        out = output.reshape(B, T, 3, n_elem, n_cls).reshape(B, T, -1, n_cls)

        def accdoa(i):
            return target[:, :, i, 0:1, :] * target[:, :, i, 1:4, :]

        perms  = self._permutations(*[accdoa(i) for i in range(6)])
        losses = torch.stack([self.mse(out, p).mean(dim=2) for p in perms])  # (13,B,T,C)
        best   = losses.min(dim=0).indices
        return sum(losses[i] * (best == i) for i in range(13)).mean()


# Alias kept for compatibility
SELDLossADPIT = ADPITLoss
