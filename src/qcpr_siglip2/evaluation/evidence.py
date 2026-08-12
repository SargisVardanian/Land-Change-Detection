"""Query-conditioned evidence diagnostics; masks are not training inputs."""

from __future__ import annotations

import torch
from torch import Tensor


def query_swap_map_l1(first_map: Tensor, second_map: Tensor) -> float:
    if first_map.shape != second_map.shape:
        raise ValueError("query maps must have equal shapes")
    return float((first_map.float() - second_map.float()).abs().mean())


def query_swap_map_cosine(first_map: Tensor, second_map: Tensor) -> float:
    if first_map.shape != second_map.shape:
        raise ValueError("query maps must have equal shapes")
    return float(
        torch.nn.functional.cosine_similarity(
            first_map.float().reshape(1, -1), second_map.float().reshape(1, -1)
        ).item()
    )


def evidence_entropy(weights: Tensor) -> Tensor:
    if weights.ndim < 1:
        raise ValueError("weights must have a token dimension")
    probabilities = weights.float().clamp_min(1e-12)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def effective_token_count(weights: Tensor) -> Tensor:
    return evidence_entropy(weights).exp()


def time_reversal_score_change(
    forward_scores: Tensor, reversed_scores: Tensor
) -> float:
    """Measure score sensitivity to swapping the temporal frame order.

    This is a diagnostic only. It does not add a direction label or a second
    training objective.
    """

    if forward_scores.shape != reversed_scores.shape:
        raise ValueError("forward and reversed scores must have equal shapes")
    if forward_scores.ndim != 2:
        raise ValueError("scores must be [queries, pairs]")
    return float((forward_scores.float() - reversed_scores.float()).abs().max())
