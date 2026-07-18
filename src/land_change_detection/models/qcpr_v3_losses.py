from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class A0ContrastiveMasks:
    text_to_pair_positive: Tensor
    text_to_pair_exclusion: Tensor
    pair_to_text_positive: Tensor
    pair_to_text_exclusion: Tensor


def a0_physical_pair_contrastive_masks(
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    pair_count: int,
) -> A0ContrastiveMasks:
    """Build physical-pair positives and duplicate-caption ambiguity masks."""
    if caption_to_pair.ndim != 1 or caption_group_ids.shape != caption_to_pair.shape:
        raise ValueError("caption_to_pair and caption_group_ids must be aligned rank-1 tensors")
    if pair_count <= 0 or caption_to_pair.numel() == 0:
        raise ValueError("non-empty mappings and positive pair_count are required")
    if int(caption_to_pair.min()) < 0 or int(caption_to_pair.max()) >= pair_count:
        raise ValueError("caption_to_pair contains an out-of-range pair index")
    query_count = caption_to_pair.numel()
    same_group = caption_group_ids[:, None] == caption_group_ids[None, :]
    text_to_pair_positive = torch.zeros(
        query_count, pair_count, dtype=torch.bool, device=caption_to_pair.device
    )
    text_to_pair_positive.scatter_(1, caption_to_pair[:, None], True)
    text_to_pair_exclusion = torch.zeros_like(text_to_pair_positive)
    for query_index in range(query_count):
        text_to_pair_exclusion[
            query_index, caption_to_pair[same_group[query_index]]
        ] = True
    text_to_pair_exclusion &= ~text_to_pair_positive

    pair_to_text_positive = torch.zeros(
        pair_count, query_count, dtype=torch.bool, device=caption_to_pair.device
    )
    pair_to_text_positive[
        caption_to_pair, torch.arange(query_count, device=caption_to_pair.device)
    ] = True
    pair_to_text_exclusion = torch.zeros_like(pair_to_text_positive)
    for pair_index in range(pair_count):
        own = caption_to_pair == pair_index
        if bool(own.any()):
            pair_to_text_exclusion[pair_index] = same_group[own].any(dim=0) & ~own
    return A0ContrastiveMasks(
        text_to_pair_positive,
        text_to_pair_exclusion,
        pair_to_text_positive,
        pair_to_text_exclusion,
    )


def a0_symmetric_physical_pair_contrastive_loss(
    scores: Tensor,
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric A0 loss; duplicate captions on other pairs are ambiguous."""
    if scores.ndim != 2:
        raise ValueError("A0 scores must have shape [captions,pairs]")
    masks = a0_physical_pair_contrastive_masks(
        caption_to_pair, caption_group_ids, pair_count=scores.shape[1]
    )
    text_to_pair = multi_positive_contrastive_loss(
        scores,
        masks.text_to_pair_positive,
        exclusion_mask=masks.text_to_pair_exclusion,
        temperature=temperature,
    )
    active_pairs = masks.pair_to_text_positive.any(dim=1)
    pair_to_text = multi_positive_contrastive_loss(
        scores.transpose(0, 1)[active_pairs],
        masks.pair_to_text_positive[active_pairs],
        exclusion_mask=masks.pair_to_text_exclusion[active_pairs],
        temperature=temperature,
    )
    return 0.5 * (text_to_pair + pair_to_text)


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
    verified_mismatched_logits: Tensor | None = None,
    mismatch_empty_weight: float = 1.0,
    generic_weight: float = 0.25,
    query_specific_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    tversky_weight: float = 1.0,
    positive_focal_weight: float = 0.75,
    negative_focal_weight: float = 0.25,
    empty_weight: float = 0.25,
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
        value = query_mask_loss(
            logits[mask], targets[mask], dice_weight=dice_weight,
            focal_weight=focal_weight, tversky_weight=tversky_weight,
            positive_focal_weight=positive_focal_weight,
            negative_focal_weight=negative_focal_weight,
            empty_weight=empty_weight,
        ).total if bool(mask.any()) else logits.sum() * 0.0
        result[f"{name}_loss"] = value
        result[f"{name}_count"] = int(mask.sum())
        # Direction is an evaluation stratum, not a second supervision target.
        if name == "generic_changed":
            total = total + float(generic_weight) * value
        elif name == "query_specific":
            total = total + float(query_specific_weight) * value
    mismatch = logits.sum() * 0.0
    if verified_mismatched_logits is not None:
        if verified_mismatched_logits.ndim != 3 or verified_mismatched_logits.shape[-2:] != logits.shape[-2:]:
            raise ValueError("verified mismatched logits must have shape [M,H,W]")
        mismatch = query_mask_loss(
            verified_mismatched_logits,
            torch.zeros_like(verified_mismatched_logits),
            dice_weight=0.0,
            focal_weight=0.0,
            tversky_weight=0.0,
        ).empty_false_positive * float(mismatch_empty_weight)
    result["verified_mismatch_count"] = 0 if verified_mismatched_logits is None else int(verified_mismatched_logits.shape[0])
    result["mismatch_provenance"] = "none" if verified_mismatched_logits is None else "explicit_recomputed_verified"
    result["generic_weight"] = float(generic_weight)
    result["query_specific_weight"] = float(query_specific_weight)
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
    tversky_weight: float = 1.0,
    boundary_weight: float = 0.0,
    empty_weight: float = 0.25,
    focal_gamma: float = 2.0,
    positive_focal_weight: float = 0.75,
    negative_focal_weight: float = 0.25,
) -> QueryMaskLossOutput:
    if min(dice_weight, focal_weight, tversky_weight, boundary_weight, empty_weight) < 0:
        raise ValueError("mask loss weights must be non-negative")
    if positive_focal_weight < 0 or negative_focal_weight < 0:
        raise ValueError("focal class weights must be non-negative")
    flat_logits, flat_targets = _flatten_masks(logits, targets)
    probabilities = flat_logits.sigmoid()
    foreground = flat_targets.sum(dim=1) > 0
    intersection = (probabilities * flat_targets).sum(dim=1)
    eps = 1e-6
    dice_each = 1.0 - (2.0 * intersection + eps) / (
        probabilities.sum(dim=1) + flat_targets.sum(dim=1) + eps
    )
    dice = dice_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
    pt = torch.where(flat_targets > 0.5, probabilities, 1.0 - probabilities)
    focal_pixels = (1.0 - pt).pow(focal_gamma) * bce
    positive_pixels = flat_targets > 0.5
    negative_pixels = ~positive_pixels
    positive_focal = (focal_pixels * positive_pixels).sum(dim=1) / positive_pixels.sum(dim=1).clamp_min(1)
    negative_focal = (focal_pixels * negative_pixels).sum(dim=1) / negative_pixels.sum(dim=1).clamp_min(1)
    # Sparse query masks need foreground-normalized gradients; otherwise the
    # background pixel count makes the all-zero solution deceptively cheap.
    focal_each = positive_focal_weight * positive_focal + negative_focal_weight * negative_focal
    focal = focal_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    false_positive = (probabilities * (1.0 - flat_targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * flat_targets).sum(dim=1)
    tversky_each = 1.0 - (intersection + eps) / (
        intersection + 0.3 * false_positive + 0.7 * false_negative + eps
    )
    tversky = tversky_each[foreground].mean() if foreground.any() else flat_logits.sum() * 0.0
    empty = ~foreground
    # Standard BCEWithLogits(z, target=0) keeps a useful gradient even for
    # confidently wrong positive logits. Empty targets are excluded from Dice.
    empty_false_positive = F.softplus(flat_logits[empty]).mean() if empty.any() else flat_logits.sum() * 0.0
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


def m0_balanced_mask_loss(logits: Tensor, targets: Tensor, *, area_weight: float = 0.1, empty_weight: float = 1.0) -> dict[str, Tensor]:
    """M0: class-normalized BCE + soft Dice + area calibration."""
    if logits.shape != targets.shape or logits.ndim != 3:
        raise ValueError("M0 logits and targets must align as [B,H,W]")
    if area_weight < 0 or empty_weight < 0:
        raise ValueError("M0 loss weights must be non-negative")
    flat_logits, flat_targets = _flatten_masks(logits, targets)
    probabilities = flat_logits.sigmoid()
    foreground = flat_targets.sum(dim=1) > 0
    empty = ~foreground
    zero = flat_logits.sum() * 0.0
    if bool(foreground.any()):
        selected_logits = flat_logits[foreground]
        selected_targets = flat_targets[foreground]
        selected_probs = probabilities[foreground]
        positive = selected_targets > 0.5
        negative = ~positive
        bce = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
        positive_bce = (bce * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        negative_bce = (bce * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1)
        balanced_bce = 0.5 * (positive_bce + negative_bce).mean()
        intersection = (selected_probs * selected_targets).sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (selected_probs.sum(dim=1) + selected_targets.sum(dim=1) + 1e-6)).mean()
        area = (selected_probs.mean(dim=1) - selected_targets.mean(dim=1)).abs().mean()
    else:
        balanced_bce = dice = area = zero
    empty_loss = F.softplus(flat_logits[empty]).mean() if bool(empty.any()) else zero
    total = balanced_bce + dice + float(area_weight) * area + float(empty_weight) * empty_loss
    return {"total": total, "balanced_bce": balanced_bce, "dice": dice, "area": area, "empty": empty_loss}


def m0_symmetric_query_swap_loss(all_logits: Tensor, mapping: Tensor, query_indices: Tensor, changes: list[str | None], targets: Tensor, *, margin: float = 0.02) -> tuple[Tensor, int]:
    """Direct appeared/disappeared contrast from independently computed logits."""
    if margin < 0 or all_logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("invalid M0 symmetric query-swap contract")
    selected = [int(value) for value in query_indices.detach().cpu().tolist()]
    lookup = {(int(mapping[q]), str(change)): local for local, (q, change) in enumerate(zip(selected, changes, strict=True)) if change in {"appeared", "disappeared"}}
    terms: list[Tensor] = []
    for direction, opposite in (("appeared", "disappeared"), ("disappeared", "appeared")):
        for pair_index, local in [(pair, local) for (pair, item_direction), local in lookup.items() if item_direction == direction]:
            other = lookup.get((pair_index, opposite))
            if other is None:
                continue
            target = targets[local].float()
            if not bool((target >= 0.5).any()):
                continue
            correct = all_logits[selected[local], pair_index].sigmoid()
            wrong = all_logits[selected[other], pair_index].sigmoid()
            def soft_iou(probability: Tensor) -> Tensor:
                intersection = (probability * target).sum()
                return (intersection + 1e-6) / (probability.sum() + target.sum() - intersection + 1e-6)
            terms.append(F.relu(float(margin) - soft_iou(correct) + soft_iou(wrong)))
    if not terms:
        return all_logits.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


def m0_aligned_symmetric_query_swap_loss(
    aligned_logits: Tensor,
    mapping: Tensor,
    changes: list[str | None],
    targets: Tensor,
    *,
    margin: float = 0.02,
) -> tuple[Tensor, int]:
    """Query-swap loss for one independently forwarded mask per query."""
    if (
        margin < 0
        or aligned_logits.ndim != 3
        or targets.shape != aligned_logits.shape
        or mapping.ndim != 1
        or mapping.numel() != aligned_logits.shape[0]
        or len(changes) != aligned_logits.shape[0]
    ):
        raise ValueError("invalid aligned M0 symmetric query-swap contract")
    lookup = {
        (int(pair), str(direction)): query
        for query, (pair, direction) in enumerate(
            zip(mapping.detach().cpu().tolist(), changes, strict=True)
        )
        if direction in {"appeared", "disappeared"}
    }
    terms: list[Tensor] = []
    for query, (pair, direction) in enumerate(
        zip(mapping.detach().cpu().tolist(), changes, strict=True)
    ):
        if direction not in {"appeared", "disappeared"}:
            continue
        opposite = "disappeared" if direction == "appeared" else "appeared"
        other = lookup.get((int(pair), opposite))
        target = targets[query].float()
        if other is None or not bool((target >= 0.5).any()):
            continue

        def soft_iou(probability: Tensor) -> Tensor:
            intersection = (probability * target).sum()
            return (intersection + 1e-6) / (
                probability.sum() + target.sum() - intersection + 1e-6
            )

        correct = soft_iou(aligned_logits[query].sigmoid())
        wrong = soft_iou(aligned_logits[other].sigmoid())
        terms.append(F.relu(float(margin) - correct + wrong))
    if not terms:
        return aligned_logits.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


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
    def safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
        return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator))
    dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    probabilities = flat_logits.sigmoid()
    foreground_probability = probabilities[truth].mean() if truth.any() else probabilities.new_zeros(())
    nonempty_background = nonempty[:, None] & ~truth
    background_probability = probabilities[nonempty_background].mean() if nonempty_background.any() else probabilities.new_zeros(())
    empty_probability = probabilities[empty].mean() if empty.any() else probabilities.new_zeros(())
    soft_intersection = (probabilities * truth.float()).sum(dim=1)
    soft_denominator = probabilities.sum(dim=1) + truth.float().sum(dim=1)
    soft_dice = (2.0 * soft_intersection + 1e-6) / (soft_denominator + 1e-6)
    soft_iou = (soft_intersection + 1e-6) / (
        probabilities.sum(dim=1) + truth.float().sum(dim=1) - soft_intersection + 1e-6
    )
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()

    # Binary average precision without interpolation. Ties are stable because
    # argsort is stable; an empty positive set is explicitly NOT_EVALUATED=0.
    labels = truth.flatten().float()
    scores = probabilities.flatten()
    if bool(labels.sum() > 0):
        order = torch.argsort(scores, descending=True, stable=True)
        ordered = labels[order]
        cumulative = ordered.cumsum(0)
        ranks = torch.arange(1, ordered.numel() + 1, device=ordered.device, dtype=ordered.dtype)
        pixel_average_precision = (cumulative.div(ranks) * ordered).sum() / labels.sum()
    else:
        pixel_average_precision = scores.new_zeros(())
    return {
        "nonempty_count": int(nonempty.sum()),
        "empty_count": int(empty.sum()),
        "nonempty_dice": mean(dice, nonempty),
        "nonempty_iou": mean(iou, nonempty),
        "nonempty_precision": mean(precision, nonempty),
        "nonempty_recall": mean(recall, nonempty),
        "micro_dice": float(safe_ratio(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn)),
        "micro_iou": float(safe_ratio(micro_tp, micro_tp + micro_fp + micro_fn)),
        "micro_precision": float(safe_ratio(micro_tp, micro_tp + micro_fp)),
        "micro_recall": float(safe_ratio(micro_tp, micro_tp + micro_fn)),
        "nonempty_soft_dice": mean(soft_dice, nonempty),
        "nonempty_soft_iou": mean(soft_iou, nonempty),
        "pixel_average_precision": float(pixel_average_precision),
        "foreground_probability": float(foreground_probability),
        "background_probability_nonempty": float(background_probability),
        "localization_margin": float(foreground_probability - background_probability),
        "empty_mean_probability": float(empty_probability),
        "samples_with_true_positive": int((tp > 0).sum()),
        "empty_false_positive_area": mean(predicted.float().mean(dim=1), empty),
        "predicted_area": float(predicted.float().mean()),
        "target_area": float(truth.float().mean()),
    }
