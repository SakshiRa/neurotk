from __future__ import annotations

from typing import Optional, Tuple

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return float((2.0 * intersection + epsilon) / (union + epsilon))


def hausdorff95(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    try:
        from monai.metrics import HausdorffDistanceMetric
    except Exception:
        return None

    pred = pred.float().unsqueeze(0).unsqueeze(0)
    target = target.float().unsqueeze(0).unsqueeze(0)
    metric = HausdorffDistanceMetric(percentile=95)
    value = metric(pred, target)
    return float(value.item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, Optional[float]]:
    d = dice_score(pred, target)
    h = hausdorff95(pred, target)
    return d, h
