from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryMaskLossOutput:
    total: Tensor
    dice: Tensor
    focal: Tensor
    tversky: Tensor
    boundary: Tensor
    empty_false_positive: Tensor


def _flatten_masks(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    if logits.shape != targets.shape or logits.ndim < 3:
        raise ValueError("logits and targets must have identical [...,H,W] shapes")
    return logits.flatten(0, -3).flatten(1), targets.float().flatten(0, -3).flatten(1)


def query_mask_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    tversky_weight: float = 0.0,
    boundary_weight: float = 0.0,
    empty_weight: float = 1.0,
    focal_gamma: float = 2.0,
) -> QueryMaskLossOutput:
    flat_logits, flat_targets = _flatten_masks(logits, targets)
    probabilities = flat_logits.sigmoid()
    foreground = flat_targets.sum(dim=1) > 0
    intersection = (probabilities * flat_targets).sum(dim=1)
    dice_each = 1.0 - (2.0 * intersection + 1.0) / (
        probabilities.sum(dim=1) + flat_targets.sum(dim=1) + 1.0
    )
    dice = dice_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
    pt = torch.where(flat_targets > 0.5, probabilities, 1.0 - probabilities)
    focal_each = ((1.0 - pt).pow(focal_gamma) * bce).mean(dim=1)
    focal = focal_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    false_positive = (probabilities * (1.0 - flat_targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * flat_targets).sum(dim=1)
    tversky_each = 1.0 - (intersection + 1.0) / (
        intersection + 0.3 * false_positive + 0.7 * false_negative + 1.0
    )
    tversky = tversky_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    empty = ~foreground
    empty_false_positive = probabilities[empty].mean() if empty.any() else flat_logits.sum() * 0.0
    boundary = flat_logits.sum() * 0.0
    if boundary_weight > 0:
        shape = logits.shape
        probs_2d = logits.sigmoid().reshape(-1, 1, shape[-2], shape[-1])
        target_2d = targets.float().reshape_as(probs_2d)
        pred_edge = torch.maximum(
            (probs_2d[..., 1:, :] - probs_2d[..., :-1, :]).abs().mean(),
            (probs_2d[..., :, 1:] - probs_2d[..., :, :-1]).abs().mean(),
        )
        target_edge = torch.maximum(
            (target_2d[..., 1:, :] - target_2d[..., :-1, :]).abs().mean(),
            (target_2d[..., :, 1:] - target_2d[..., :, :-1]).abs().mean(),
        )
        boundary = (pred_edge - target_edge).abs()
    total = (
        dice_weight * dice + focal_weight * focal + tversky_weight * tversky
        + boundary_weight * boundary + empty_weight * empty_false_positive
    )
    return QueryMaskLossOutput(total, dice, focal, tversky, boundary, empty_false_positive)


def query_mask_metrics(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> dict[str, float | int]:
    flat_logits, flat_targets = _flatten_masks(logits, targets)
    predicted = flat_logits.sigmoid() >= threshold
    truth = flat_targets >= 0.5
    nonempty = truth.any(dim=1)
    empty = ~nonempty
    tp = (predicted & truth).sum(dim=1).float()
    fp = (predicted & ~truth).sum(dim=1).float()
    fn = (~predicted & truth).sum(dim=1).float()
    def mean(values: Tensor, mask: Tensor) -> float:
        return float(values[mask].mean()) if mask.any() else 0.0
    dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    iou = (tp + 1) / (tp + fp + fn + 1)
    precision = (tp + 1) / (tp + fp + 1)
    recall = (tp + 1) / (tp + fn + 1)
    return {
        "nonempty_count": int(nonempty.sum()),
        "empty_count": int(empty.sum()),
        "nonempty_dice": mean(dice, nonempty),
        "nonempty_iou": mean(iou, nonempty),
        "nonempty_precision": mean(precision, nonempty),
        "nonempty_recall": mean(recall, nonempty),
        "empty_false_positive_area": mean(predicted.float().mean(dim=1), empty),
        "predicted_area": float(predicted.float().mean()),
        "target_area": float(truth.float().mean()),
    }
