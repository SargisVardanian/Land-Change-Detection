"""Protocol-explicit retrieval metrics for the QCPR benchmark.

The functions operate on ranked physical IDs and relevance sets; no model or
dataset-specific assumptions are hidden in the implementation.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    for index, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def recall_at(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(bool(set(ranking[:k]) & relevant))


def ndcg_at(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / __import__("math").log2(index + 2) for index, item in enumerate(ranking[:k]) if item in relevant)
    ideal = sum(1.0 / __import__("math").log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def aggregate_rankings(rows: Iterable[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    values = list(rows)
    ranks = [reciprocal_rank(ranking, relevant) for ranking, relevant in values]
    return {
        "queries": len(values),
        "mrr": sum(ranks) / len(ranks) if ranks else 0.0,
        "recall@1": sum(recall_at(r, rel, 1) for r, rel in values) / len(values) if values else 0.0,
        "recall@5": sum(recall_at(r, rel, 5) for r, rel in values) / len(values) if values else 0.0,
        "recall@10": sum(recall_at(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
        "ndcg@10": sum(ndcg_at(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
    }


def changeretcap_compat(ranking: Sequence[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    """Top-k compatibility view; it is not interchangeable with full-gallery MRR."""
    top = list(ranking[:k])
    return {"k": k, "precision": len(set(top) & relevant) / k, "recall": float(bool(set(top) & relevant)), "mrr@k": reciprocal_rank(top, relevant)}
