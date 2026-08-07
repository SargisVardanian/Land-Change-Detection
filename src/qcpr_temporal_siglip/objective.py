from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


RelevanceMode = Literal["exact", "semantic"]


@dataclass(frozen=True)
class RelevanceMasks:
    positive: Tensor
    ignored: Tensor


def build_relevance_masks(
    graded_relevance: Tensor,
    *,
    mode: RelevanceMode = "exact",
    include_grade1: bool = False,
) -> RelevanceMasks:
    """Convert verified grades to compact positive/ignored masks.

    Event IDs and identical text are deliberately absent from this function.
    The caller must supply reviewed relevance grades.
    """

    if graded_relevance.ndim != 2:
        raise ValueError("graded_relevance must be [queries, pairs]")
    if not bool(((graded_relevance >= 0) & (graded_relevance <= 3)).all()):
        raise ValueError("relevance grades must be integers in [0, 3]")
    if mode == "exact":
        positive = graded_relevance == 3
        ignored = (graded_relevance == 1) | (graded_relevance == 2)
    elif mode == "semantic":
        positive = (graded_relevance == 3) | (graded_relevance == 2)
        ignored = graded_relevance == 1
        if include_grade1:
            positive = positive | (graded_relevance == 1)
            ignored = torch.zeros_like(positive)
    else:
        raise ValueError(f"unsupported relevance mode: {mode}")
    if bool((positive & ignored).any()):
        raise AssertionError("positive and ignored relevance overlap")
    return RelevanceMasks(positive=positive.bool(), ignored=ignored.bool())


def positive_weights_from_grades(
    graded_relevance: Tensor,
    masks: RelevanceMasks,
    *,
    grade1_weight: float = 0.25,
) -> Tensor:
    """Return optional numerator weights without creating new positives.

    Grade 3 and grade 2 positives receive weight one.  Grade 1 can receive a
    small weight only when it was explicitly enabled in the semantic masks.
    """

    if graded_relevance.shape != masks.positive.shape:
        raise ValueError("graded_relevance must match relevance masks")
    if grade1_weight < 0.0:
        raise ValueError("grade1_weight must be non-negative")
    weights = torch.ones_like(graded_relevance, dtype=torch.float32)
    weights = torch.where(
        graded_relevance == 1,
        torch.full_like(weights, grade1_weight),
        weights,
    )
    weights = torch.where(masks.positive, weights, torch.ones_like(weights))
    return weights


def _direction_loss(
    scores: Tensor,
    positive: Tensor,
    ignored: Tensor,
    *,
    grade_weights: Tensor | None = None,
) -> Tensor:
    if scores.ndim != 2 or positive.shape != scores.shape or ignored.shape != scores.shape:
        raise ValueError("scores and relevance masks must have the same [Q,P] shape")
    if bool((positive & ignored).any()):
        raise ValueError("positive and ignored masks overlap")
    if bool((positive.sum(dim=1) == 0).any()):
        raise ValueError("every row must have at least one positive")
    valid = ~ignored
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("every row must have at least one valid candidate")
    scores_fp32 = scores.float()
    masked_positive = scores_fp32.masked_fill(~positive, float("-inf"))
    masked_valid = scores_fp32.masked_fill(~valid, float("-inf"))
    if grade_weights is None:
        numerator = torch.logsumexp(masked_positive, dim=1)
    else:
        if grade_weights.shape != scores.shape:
            raise ValueError("grade_weights must match scores")
        numerator = torch.logsumexp(
            masked_positive + grade_weights.float().clamp_min(torch.finfo(torch.float32).tiny).log(),
            dim=1,
        )
    return -(numerator - torch.logsumexp(masked_valid, dim=1)).mean()


def symmetric_mult_positive_clip_loss(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor | None = None,
    *,
    positive_weights: Tensor | None = None,
) -> Tensor:
    """One symmetric multi-positive CLIP objective.

    The result is exactly one scalar:

    0.5 * text-to-pair + 0.5 * pair-to-text.
    """

    if ignored_mask is None:
        ignored_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
    if scores.ndim != 2 or positive_mask.shape != scores.shape or ignored_mask.shape != scores.shape:
        raise ValueError("scores and masks must have shape [queries, pairs]")
    text_to_pair = _direction_loss(
        scores,
        positive_mask.bool(),
        ignored_mask.bool(),
        grade_weights=positive_weights,
    )
    pair_to_text = _direction_loss(
        scores.transpose(0, 1),
        positive_mask.transpose(0, 1).bool(),
        ignored_mask.transpose(0, 1).bool(),
        grade_weights=positive_weights.transpose(0, 1) if positive_weights is not None else None,
    )
    loss = 0.5 * (text_to_pair + pair_to_text)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("symmetric CLIP loss is non-finite")
    return loss
