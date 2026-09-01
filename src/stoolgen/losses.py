"""Geometry-aware objectives for unordered point sets."""

import torch


def chamfer_distance(prediction: torch.Tensor, target: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Symmetric squared Chamfer distance with chunks to limit peak memory."""
    pred_to_target = []
    for start in range(0, prediction.shape[1], chunk):
        distances = torch.cdist(prediction[:, start:start + chunk], target).square()
        pred_to_target.append(distances.min(dim=2).values)
    target_to_pred = []
    for start in range(0, target.shape[1], chunk):
        distances = torch.cdist(target[:, start:start + chunk], prediction).square()
        target_to_pred.append(distances.min(dim=2).values)
    return torch.cat(pred_to_target, 1).mean() + torch.cat(target_to_pred, 1).mean()


def kl_divergence(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=1).mean()

