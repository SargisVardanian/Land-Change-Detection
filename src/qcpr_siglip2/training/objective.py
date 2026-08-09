from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..contracts import validate_relevance_masks


def multi_positive_listwise_loss(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor | None = None,
    graded_relevance: Tensor | None = None,
    *,
    grade_weights: dict[int, float] | None = None,
) -> Tensor:
    """One scalar multi-positive listwise retrieval objective."""
    if scores.ndim != 2:
        raise ValueError("scores must be [queries, pairs]")
    if ignored_mask is None:
        ignored_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
    validate_relevance_masks(
        positive_mask, ignored_mask, score_shape=tuple(scores.shape)
    )
    if graded_relevance is not None and graded_relevance.shape != scores.shape:
        raise ValueError("graded_relevance must match scores")
    weights = torch.ones_like(scores)
    if graded_relevance is not None:
        mapping = grade_weights or {1: 0.25, 2: 0.75, 3: 1.0}
        weights = torch.zeros_like(scores)
        for grade, weight in mapping.items():
            weights = torch.where(
                graded_relevance == grade,
                torch.as_tensor(weight, device=scores.device, dtype=scores.dtype),
                weights,
            )
        weights = torch.where(
            positive_mask, weights.clamp_min(torch.finfo(scores.dtype).tiny), weights
        )
    valid_scores = scores.masked_fill(ignored_mask, torch.finfo(scores.dtype).min)
    positive_scores = scores.masked_fill(~positive_mask, torch.finfo(scores.dtype).min)
    numerator = torch.logsumexp(
        positive_scores + weights.clamp_min(torch.finfo(scores.dtype).tiny).log(), dim=1
    )
    denominator = torch.logsumexp(valid_scores, dim=1)
    loss = -(numerator - denominator).mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("listwise loss is non-finite")
    return loss


def symmetric_multi_positive_listwise_loss(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor | None = None,
) -> Tensor:
    """One scalar loss covering text-to-pair and pair-to-text relevance."""

    if ignored_mask is None:
        ignored_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
    query_to_pair = multi_positive_listwise_loss(
        scores, positive_mask, ignored_mask
    )
    if torch.any(positive_mask.sum(dim=0) == 0):
        raise ValueError("every physical pair needs at least one positive text")
    pair_to_text = multi_positive_listwise_loss(
        scores.transpose(0, 1),
        positive_mask.transpose(0, 1),
        ignored_mask.transpose(0, 1),
    )
    loss = 0.5 * (query_to_pair + pair_to_text)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("symmetric listwise loss is non-finite")
    return loss


def listwise_loss_diagnostics(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor | None = None,
    graded_relevance: Tensor | None = None,
) -> dict[str, Any]:
    loss = multi_positive_listwise_loss(
        scores, positive_mask, ignored_mask, graded_relevance
    )
    return {
        "loss": float(loss.detach().cpu()),
        "score_shape": [int(x) for x in scores.shape],
        "positive_count": int(positive_mask.sum()),
        "ignored_count": int(
            (
                ignored_mask
                if ignored_mask is not None
                else torch.zeros_like(positive_mask)
            ).sum()
        ),
    }
