from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def duplicate_aware_positive_mask(
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    pair_count: int,
) -> Tensor:
    """Map exact and normalized-caption-equivalent queries to positive pairs."""
    if caption_to_pair.ndim != 1 or caption_group_ids.shape != caption_to_pair.shape:
        raise ValueError("caption_to_pair and caption_group_ids must be aligned rank-1 tensors")
    if pair_count <= 0 or caption_to_pair.numel() == 0:
        raise ValueError("non-empty mappings and positive pair_count are required")
    if int(caption_to_pair.min()) < 0 or int(caption_to_pair.max()) >= pair_count:
        raise ValueError("caption_to_pair contains an out-of-range pair index")
    same_group = caption_group_ids[:, None] == caption_group_ids[None, :]
    positives = torch.zeros(
        caption_to_pair.numel(), pair_count, dtype=torch.bool, device=caption_to_pair.device
    )
    for caption_index in range(caption_to_pair.numel()):
        positives[:, caption_to_pair[caption_index]] |= same_group[:, caption_index]
    positives.scatter_(1, caption_to_pair[:, None], True)
    return positives


def multi_positive_contrastive_loss(
    scores: Tensor,
    positive_mask: Tensor,
    *,
    exclusion_mask: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    """Set-valued contrastive loss with optional false-negative exclusion."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have the same rank-2 shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positives = positive_mask.bool()
    if not bool(positives.any(dim=1).all()):
        raise ValueError("every query must have at least one positive")
    valid = torch.ones_like(positives)
    if exclusion_mask is not None:
        if exclusion_mask.shape != scores.shape:
            raise ValueError("exclusion_mask must match scores")
        valid = ~exclusion_mask.bool()
        valid |= positives
    scaled = scores.float() / float(temperature)
    numerator = torch.logsumexp(scaled.masked_fill(~positives, float("-inf")), dim=1)
    denominator = torch.logsumexp(scaled.masked_fill(~valid, float("-inf")), dim=1)
    loss = (denominator - numerator).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("multi-positive contrastive loss is non-finite")
    return loss


@dataclass(frozen=True)
class QueryMaskLossOutput:
    total: Tensor
    dice: Tensor
    focal: Tensor
    tversky: Tensor
    boundary: Tensor
    empty_false_positive: Tensor




def foreground_preserving_resize(targets: Tensor, size: tuple[int, int]) -> Tensor:
    """Downsample masks with max pooling so a small foreground cannot vanish."""
    if targets.ndim != 3:
        raise ValueError("targets must have shape [B,H,W]")
    height, width = targets.shape[-2:]
    if size[0] <= height and size[1] <= width:
        return F.adaptive_max_pool2d(targets.float().unsqueeze(1), size)[:, 0]
    return F.interpolate(targets.float().unsqueeze(1), size=size, mode="nearest")[:, 0]


def separated_query_mask_losses(
    logits: Tensor,
    targets: Tensor,
    target_kinds: list[str],
    change_types: list[str | None],
    *,
    mismatch_empty_weight: float = 1.0,
) -> dict[str, Tensor | int]:
    """Separate mask losses by provenance and add mismatched-query empty negatives.

    Inputs are already paired query logits.  Parser labels choose reporting and
    supervision provenance only; they never create semantic model heads.
    """
    if logits.ndim != 3 or targets.shape != logits.shape or len(target_kinds) != logits.shape[0] or len(change_types) != logits.shape[0]:
        raise ValueError("paired logits/targets and metadata must align")
    def pick(predicate):
        return torch.tensor([predicate(kind, change) for kind, change in zip(target_kinds, change_types, strict=True)], device=logits.device, dtype=torch.bool)
    groups = {
        "generic_changed": pick(lambda kind, change: kind in {"binary_generic", "semantic_transition_union"}),
        "query_specific": pick(lambda kind, change: kind == "query_specific"),
        "appeared": pick(lambda kind, change: change == "appeared"),
        "disappeared": pick(lambda kind, change: change == "disappeared"),
    }
    result: dict[str, Tensor | int] = {}
    total = logits.sum() * 0.0
    for name, mask in groups.items():
        value = query_mask_loss(logits[mask], targets[mask]).total if bool(mask.any()) else logits.sum() * 0.0
        result[f"{name}_loss"] = value
        result[f"{name}_count"] = int(mask.sum())
        total = total + value
    query_mask = groups["query_specific"]
    mismatch = logits.sum() * 0.0
    if int(query_mask.sum()) > 1:
        selected = logits[query_mask]
        mismatched = selected.roll(shifts=1, dims=0)
        mismatch = query_mask_loss(mismatched, torch.zeros_like(mismatched)).empty_false_positive * float(mismatch_empty_weight)
    result["mismatched_query_empty_loss"] = mismatch
    result["total"] = total + mismatch
    return result

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
