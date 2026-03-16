import torch
import torch.nn as nn


class SELDLoss(nn.Module):
    """Combined loss for single ACCDOA SELD output.

    L = alpha * L_doa + beta * L_dist

    L_doa:  Mean Squared Error on ACCDOA DOA vectors (act*x, act*y)
    L_dist: Mean Squared Percentage Error on source distances, masked
            by ground-truth activity indicator

    Reference: Wu & Zhu, DCASE2025 Technical Report, Equations (4)-(6)
    """

    def __init__(self, nb_classes=13, alpha=1.0, beta=2.0):
        super().__init__()
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, T, 3*C) predicted single ACCDOA [act*x, act*y, dist]
            target: (B, T, 3*C) ground truth single ACCDOA
        Returns:
            loss: scalar combined loss
        """
        C = self.nb_classes

        # --- DOA loss: MSE on activity-coupled direction vectors ---
        pred_doa = pred[:, :, :2 * C]
        target_doa = target[:, :, :2 * C]
        loss_doa = torch.mean((pred_doa - target_doa) ** 2)

        # --- Distance loss: MSPE on active events only ---
        pred_dist = pred[:, :, 2 * C:]     # (B, T, C)
        target_dist = target[:, :, 2 * C:]  # (B, T, C)

        # Derive ground-truth activity from target ACCDOA magnitude
        target_x = target[:, :, :C]
        target_y = target[:, :, C:2 * C]
        activity = (target_x ** 2 + target_y ** 2).sqrt() > 0.5  # (B, T, C) bool

        # Compute MSPE: L_dist = mean( (a * (d_gt - d_pred) / d_gt)^2 )
        # For inactive events (a=0), contribution is zero.
        # For active events, compute squared relative error.
        eps = 1e-7
        activity_f = activity.float()
        relative_error = activity_f * (target_dist - pred_dist) / (target_dist + eps)
        loss_dist = torch.mean(relative_error ** 2)

        loss = self.alpha * loss_doa + self.beta * loss_dist
        return loss


if __name__ == '__main__':
    criterion = SELDLoss(nb_classes=13, alpha=1.0, beta=2.0)

    pred = torch.randn(4, 50, 39)
    target = torch.randn(4, 50, 39)
    loss = criterion(pred, target)
    print(f'Loss: {loss.item():.4f}')
