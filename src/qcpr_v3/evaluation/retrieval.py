"""Common-gallery retrieval and candidate evaluation."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def rank_scores(scores: Tensor) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [Q,P]")
    return scores.argsort(dim=-1, descending=True, stable=True)



def _validate_relevance(scores: Tensor, relevant: Tensor) -> None:
    if scores.shape != relevant.shape or relevant.dtype != torch.bool:
        raise ValueError("scores and relevant must share [Q,P] and relevant must be bool")
    if not bool(relevant.any(dim=1).all()):
        raise ValueError("every query must have at least one relevant item")


def _topk_relevance(order: Tensor, relevant: Tensor, k: int) -> Tensor:
    clipped = min(int(k), order.shape[1])
    return relevant.gather(1, order[:, :clipped])


def _first_relevant_rank(order: Tensor, relevant: Tensor) -> Tensor:
    gathered = relevant.gather(1, order)
    positions = torch.arange(1, order.shape[1] + 1, device=order.device).view(1, -1)
    sentinel = order.shape[1] + 1
    return torch.where(gathered, positions, torch.full_like(positions, sentinel)).amin(dim=1)


def hit_rate_at_k(scores: Tensor, relevant: Tensor, k: int) -> float:
    _validate_relevance(scores, relevant)
    return float(_topk_relevance(rank_scores(scores), relevant, k).any(dim=1).float().mean())


def recall_at_k(scores: Tensor, relevant: Tensor, k: int) -> float:
    _validate_relevance(scores, relevant)
    top = _topk_relevance(rank_scores(scores), relevant, k).sum(dim=1).float()
    total = relevant.sum(dim=1).float()
    return float((top / total.clamp_min(1.0)).mean())


def precision_at_k(scores: Tensor, relevant: Tensor, k: int) -> float:
    _validate_relevance(scores, relevant)
    top = _topk_relevance(rank_scores(scores), relevant, k).sum(dim=1).float()
    return float((top / min(int(k), scores.shape[1])).mean())


def mrr_at_k(scores: Tensor, relevant: Tensor, k: int) -> float:
    _validate_relevance(scores, relevant)
    ranks = _first_relevant_rank(rank_scores(scores), relevant)
    return float(torch.where(ranks <= int(k), 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float32)).mean())


def mean_average_precision(scores: Tensor, relevant: Tensor) -> float:
    """Full-gallery mean average precision for binary relevance."""
    _validate_relevance(scores, relevant)
    ordered = relevant.gather(1, rank_scores(scores))
    positions = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=torch.float32).view(1, -1)
    precision = ordered.cumsum(dim=1).float() / positions
    average_precision = (precision * ordered.float()).sum(dim=1) / relevant.sum(dim=1).float().clamp_min(1.0)
    return float(average_precision.mean())


def ndcg_at_k(scores: Tensor, graded_relevance: Tensor, k: int) -> float:
    if scores.shape != graded_relevance.shape:
        raise ValueError("scores and graded_relevance must share [Q,P]")
    order = rank_scores(scores)
    clipped = min(int(k), scores.shape[1])
    gains = graded_relevance.gather(1, order[:, :clipped]).float()
    discounts = 1.0 / torch.log2(torch.arange(2, clipped + 2, device=scores.device, dtype=torch.float32))
    dcg = ((2.0 ** gains - 1.0) * discounts).sum(dim=1)
    ideal = torch.sort(graded_relevance.float(), dim=1, descending=True).values[:, :clipped]
    idcg = ((2.0 ** ideal - 1.0) * discounts).sum(dim=1)
    return float((dcg / idcg.clamp_min(1e-8)).mean())


def retrieval_metrics(scores: Tensor, relevant: Tensor, ks: Iterable[int] = (1, 5, 10, 50, 100, 500)) -> dict[str, float]:
    _validate_relevance(scores, relevant)
    order = rank_scores(scores)
    ranks = _first_relevant_rank(order, relevant)
    result: dict[str, float] = {
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.median()),
        "mean_average_precision": mean_average_precision(scores, relevant),
    }
    for k in ks:
        result[f"hit_rate_at_{k}"] = hit_rate_at_k(scores, relevant, int(k))
        result[f"recall_at_{k}"] = recall_at_k(scores, relevant, int(k))
        result[f"precision_at_{k}"] = precision_at_k(scores, relevant, int(k))
        result[f"mrr_at_{k}"] = mrr_at_k(scores, relevant, int(k))
        result[f"ndcg_at_{k}"] = ndcg_at_k(scores, relevant.to(dtype=torch.float32), int(k))
    return result


def pair_to_text_metrics(scores: Tensor, relevant: Tensor, ks: Iterable[int] = (1, 5, 10, 50, 100)) -> dict[str, float]:
    """Evaluate the reverse direction using the same explicit relevance contract."""
    if scores.ndim != 2 or relevant.shape != scores.shape:
        raise ValueError("scores and relevant must share [queries, items]")
    return retrieval_metrics(scores.transpose(0, 1), relevant.transpose(0, 1), ks=ks)


def candidate_recall(scores: Tensor, relevant: Tensor, k: int) -> float:
    return hit_rate_at_k(scores, relevant, k)
