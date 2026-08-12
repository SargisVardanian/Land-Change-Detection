from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..contracts import validate_relevance_masks


def _multi_positive_listwise_loss_per_row(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor,
    graded_relevance: Tensor | None = None,
    *,
    grade_weights: dict[int, float] | None = None,
) -> Tensor:
    """Return one finite listwise loss for every query/candidate row.

    This is the unreduced form used by the variable-query objective.  Keeping
    the reduction explicit is important: averaging over query rows would give
    a physical pair with two captions twice the text-to-pair weight of a pair
    with one caption.
    """

    if scores.ndim != 2:
        raise ValueError("scores must be [rows, candidates]")
    validate_relevance_masks(
        positive_mask, ignored_mask, score_shape=tuple(scores.shape)
    )
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("every row needs at least one positive")
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
    losses = -(numerator - denominator)
    if losses.ndim != 1 or not torch.isfinite(losses).all():
        raise FloatingPointError("listwise loss is non-finite")
    return losses


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
    loss = _multi_positive_listwise_loss_per_row(
        scores,
        positive_mask,
        ignored_mask,
        graded_relevance,
        grade_weights=grade_weights,
    ).mean()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("listwise loss is non-finite")
    return loss


def pair_balanced_text_to_pair_loss(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor,
    query_pair_indices: Tensor,
    *,
    graded_relevance: Tensor | None = None,
    grade_weights: dict[int, float] | None = None,
) -> Tensor:
    """Text-to-pair loss with equal aggregate weight for every physical pair.

    ``query_pair_indices[q]`` identifies the physical-pair row associated with
    query ``q``.  Query rows remain unique; a pair with several captions gets
    the mean of its query losses, then every pair contributes one equally
    weighted term.
    """

    if query_pair_indices.ndim != 1 or query_pair_indices.numel() != scores.shape[0]:
        raise ValueError("query_pair_indices must have one entry per query row")
    if scores.shape[1] <= 0:
        raise ValueError("text-to-pair loss requires at least one physical pair")
    if query_pair_indices.device != scores.device:
        query_pair_indices = query_pair_indices.to(scores.device)
    if query_pair_indices.dtype != torch.long:
        query_pair_indices = query_pair_indices.long()
    if torch.any(query_pair_indices < 0) or torch.any(
        query_pair_indices >= scores.shape[1]
    ):
        raise ValueError("query_pair_indices contains an invalid physical-pair index")
    row_losses = _multi_positive_listwise_loss_per_row(
        scores,
        positive_mask,
        ignored_mask,
        graded_relevance,
        grade_weights=grade_weights,
    )
    pair_losses: list[Tensor] = []
    for pair_index in range(scores.shape[1]):
        member = query_pair_indices == pair_index
        if not torch.any(member):
            raise ValueError(f"physical pair {pair_index} has no sampled query")
        pair_losses.append(row_losses[member].mean())
    return torch.stack(pair_losses).mean()


def pair_balanced_symmetric_multi_positive_listwise_loss(
    scores: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor | None,
    query_pair_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """The single symmetric variable-query objective.

    Returns ``(total, text_to_pair, pair_to_text)`` for auditable loss
    accounting.  Pair-to-text uses the complete set of unique query rows for
    each physical pair as one multi-positive numerator.
    """

    if ignored_mask is None:
        ignored_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
    if positive_mask.ndim != 2:
        raise ValueError("positive_mask must be [queries, pairs]")
    if torch.any(positive_mask.sum(dim=0) == 0):
        raise ValueError("every physical pair needs at least one positive text")
    text_to_pair = pair_balanced_text_to_pair_loss(
        scores, positive_mask, ignored_mask, query_pair_indices
    )
    pair_to_text = multi_positive_listwise_loss(
        scores.transpose(0, 1),
        positive_mask.transpose(0, 1),
        ignored_mask.transpose(0, 1),
    )
    loss = 0.5 * (text_to_pair + pair_to_text)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("listwise loss is non-finite")
    return loss, text_to_pair, pair_to_text


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
