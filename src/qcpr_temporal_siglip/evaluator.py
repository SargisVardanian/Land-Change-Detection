from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def rank_pairs(scores: Tensor, *, top_k: int = 10) -> Tensor:
    """Return direct pair indices; no reranking or candidate cascade."""

    if scores.ndim != 2:
        raise ValueError("scores must be [queries, pairs]")
    if top_k <= 0 or top_k > scores.shape[1]:
        raise ValueError("top_k must be within the pair gallery")
    return scores.topk(top_k, dim=1, largest=True, sorted=True).indices


def _ranks(scores: Tensor, relevant: Tensor) -> Tensor:
    if scores.shape != relevant.shape or scores.ndim != 2:
        raise ValueError("scores and relevance must have the same [Q,P] shape")
    if bool((relevant.sum(dim=1) == 0).any()):
        raise ValueError("every evaluated query needs at least one relevant pair")
    order = scores.argsort(dim=1, descending=True)
    ordered_relevance = relevant.gather(1, order)
    first = ordered_relevance.float().argmax(dim=1)
    return first + 1


def _dcg(values: Tensor) -> Tensor:
    positions = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=torch.float32)
    return ((2.0**values.float() - 1.0) / torch.log2(positions + 1.0)).sum(dim=-1)


def direct_retrieval_metrics(
    scores: Tensor,
    relevant: Tensor,
    *,
    semantic_relevance: Tensor | None = None,
) -> dict[str, float]:
    """Primary direct-retrieval dashboard with explicit metric names."""

    ranks = _ranks(scores, relevant)
    result: dict[str, float] = {
        "hit_at_1": float((ranks <= 1).float().mean()),
        "hit_at_5": float((ranks <= 5).float().mean()),
        "hit_at_10": float((ranks <= 10).float().mean()),
        "hit_at_100": float((ranks <= 100).float().mean()),
        "mrr_full": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": float(ranks.float().median()),
    }
    if semantic_relevance is not None:
        if semantic_relevance.shape != scores.shape:
            raise ValueError("semantic_relevance must match scores")
        top10 = rank_pairs(scores, top_k=min(10, scores.shape[1]))
        semantic_binary = semantic_relevance > 0
        semantic_top = semantic_binary.gather(1, top10)
        relevant_count = semantic_binary.sum(dim=1).clamp_min(1)
        result["semantic_recall_at_10"] = float(
            (semantic_top.sum(dim=1) / relevant_count).mean()
        )
        grades = semantic_relevance.float().gather(1, top10)
        ideal_count = min(10, scores.shape[1])
        ideal = semantic_relevance.float().topk(ideal_count, dim=1).values
        result["ndcg_at_10"] = float(
            (_dcg(grades) / _dcg(ideal).clamp_min(1e-8)).mean()
        )
    return result


def metric_contract() -> dict[str, Any]:
    return {
        "primary_inference": "direct query-to-pair cosine ranking",
        "primary_k": 10,
        "metrics": {
            "hit_at_10": "fraction of queries with at least one exact relevant pair in Top-10",
            "semantic_recall_at_10": "number of semantic relevant pairs in Top-10 divided by total semantic relevant pairs",
            "ndcg_at_10": "graded relevance nDCG over direct Top-10 ranking",
            "mrr_full": "mean reciprocal rank of first exact relevant pair over full gallery",
        },
        "reranking": "not part of the active TemporalSigLIP path",
    }
