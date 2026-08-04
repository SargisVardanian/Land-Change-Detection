"""Common-gallery retrieval and candidate evaluation."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def rank_scores(scores: Tensor) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [Q,P]")
    return scores.argsort(dim=-1, descending=True, stable=True)


def _first_relevant_rank(order: Tensor, relevant: Tensor) -> Tensor:
    q, p = order.shape
    gathered = relevant.gather(1, order)
    positions = torch.arange(1, p + 1, device=order.device).view(1, -1)
    sentinel = p + 1
    return torch.where(gathered, positions, torch.full_like(positions, sentinel)).amin(dim=1)


def retrieval_metrics(scores: Tensor, relevant: Tensor, ks: Iterable[int] = (1, 5, 10, 50, 100, 500)) -> dict[str, float]:
    if scores.shape != relevant.shape or relevant.dtype != torch.bool:
        raise ValueError("scores and relevant must share [Q,P] and relevant must be bool")
    order = rank_scores(scores)
    ranks = _first_relevant_rank(order, relevant)
    result: dict[str, float] = {
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.median()),
    }
    for k in ks:
        clipped = min(int(k), scores.shape[1])
        hits = order[:, :clipped]
        result[f"recall_at_{k}"] = float(relevant.gather(1, hits).any(dim=1).float().mean())
    return result


def candidate_recall(scores: Tensor, relevant: Tensor, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    return float(relevant.gather(1, rank_scores(scores)[:, : min(k, scores.shape[1])]).any(dim=1).float().mean())
