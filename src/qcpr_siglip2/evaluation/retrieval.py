from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def first_positive_ranks(scores: Tensor, relevance: Tensor) -> Tensor:
    """Return one-based first-positive ranks after exactly one gallery sort."""

    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have equal [Q,P] shapes")
    ranked = relevance.gather(1, scores.argsort(dim=1, descending=True))
    first = ranked.float().argmax(dim=1) + 1
    missing = ranked.sum(dim=1) == 0
    return torch.where(missing, torch.full_like(first, scores.shape[1] + 1), first)


def candidate_hit_at_k(scores: Tensor, relevance: Tensor, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    order = scores.argsort(dim=1, descending=True)[:, :k]
    return float(relevance.gather(1, order).any(dim=1).float().mean())


def multi_positive_recall_at_k(scores: Tensor, relevance: Tensor, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    order = scores.argsort(dim=1, descending=True)[:, :k]
    hits = relevance.gather(1, order).sum(dim=1).float()
    total = relevance.sum(dim=1).float().clamp_min(1.0)
    return float((hits / total).mean())


def precision_at_k(scores: Tensor, relevance: Tensor, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    order = scores.argsort(dim=1, descending=True)[:, :k]
    return float(relevance.gather(1, order).sum(dim=1).float().div(float(k)).mean())


def mrr_full(scores: Tensor, relevance: Tensor) -> float:
    return float((1.0 / first_positive_ranks(scores, relevance).float()).mean())


def mean_rank(scores: Tensor, relevance: Tensor) -> float:
    return float(first_positive_ranks(scores, relevance).float().mean())


def median_rank(scores: Tensor, relevance: Tensor) -> float:
    return float(first_positive_ranks(scores, relevance).float().median())


def mrr_at_k(scores: Tensor, relevance: Tensor, k: int) -> float:
    ranks = first_positive_ranks(scores, relevance)
    return float(
        torch.where(
            ranks <= k,
            1.0 / ranks.float(),
            torch.zeros_like(ranks, dtype=torch.float32),
        ).mean()
    )


def map_at_k(scores: Tensor, relevance: Tensor, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    order = scores.argsort(dim=1, descending=True)[:, :k]
    ranked = relevance.gather(1, order).float()
    cumulative = ranked.cumsum(1)
    pos = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32).view(1, -1)
    denom = relevance.sum(1).float().clamp_min(1.0)
    return float(((cumulative / pos * ranked).sum(1) / denom).mean())


def full_gallery_metrics(
    scores: Tensor, relevance: Tensor, ks: Iterable[int] = (1, 5, 10, 50, 100, 500)
) -> dict[str, float]:
    """Compute all exact/multi-positive metrics from one deterministic sort.

    Earlier code called every public metric independently, causing one full
    gallery ``argsort`` per metric.  This fused implementation preserves the
    formulas while making full-gallery evaluation and clustered bootstrap
    practical.
    """

    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have equal [Q,P] shapes")
    order = scores.argsort(dim=1, descending=True)
    ranked = relevance.gather(1, order)
    first = ranked.float().argmax(dim=1) + 1
    missing = ranked.sum(dim=1) == 0
    ranks = torch.where(
        missing, torch.full_like(first, scores.shape[1] + 1), first
    )
    result = {
        "mrr_full": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }
    for k in ks:
        kk = min(int(k), scores.shape[1])
        top = ranked[:, :kk].float()
        hits = top.sum(dim=1)
        total = relevance.sum(dim=1).float().clamp_min(1.0)
        positions = torch.arange(
            1, kk + 1, device=scores.device, dtype=torch.float32
        ).view(1, -1)
        result[f"candidate_hit_at_{k}"] = float(top.bool().any(dim=1).float().mean())
        result[f"multi_positive_recall_at_{k}"] = float((hits / total).mean())
        result[f"precision_at_{k}"] = float(hits.div(float(kk)).mean())
        result[f"mrr_at_{k}"] = float(
            torch.where(
                ranks <= kk,
                1.0 / ranks.float(),
                torch.zeros_like(ranks, dtype=torch.float32),
            ).mean()
        )
        cumulative = top.cumsum(1)
        result[f"map_at_{k}"] = float(
            ((cumulative / positions * top).sum(1) / total).mean()
        )
    return result
