"""Common exact-gallery evaluation for global retrieval and Top-K reranking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..data.manifest import group_rows_by_pair
from .reranking import select_topk_candidates
from .retrieval import full_gallery_metrics


def _sequence_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_pair_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic image row per physical pair."""

    grouped = group_rows_by_pair(rows)
    result = [group[0] for group in grouped.values()]
    if not result:
        raise ValueError("common gallery has no physical pairs")
    return result


def exact_relevance_masks(
    query_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Build exact masks without using caption text or implicit collisions.

    Older exact manifests identify the positive by ``canonical_pair_id`` while
    newer rows may carry an explicit ``positive_pair_ids`` list.  Both are
    accepted; no text-based collision inference is performed.
    """

    target_device = device or torch.device("cpu")
    pair_index = {
        str(row["canonical_pair_id"]): index
        for index, row in enumerate(pair_rows)
    }
    positive = torch.zeros(
        (len(query_rows), len(pair_rows)), dtype=torch.bool, device=target_device
    )
    ignored = torch.zeros_like(positive)
    for query_index, row in enumerate(query_rows):
        raw_positive = row.get("positive_pair_ids")
        positive_ids = (
            raw_positive
            if isinstance(raw_positive, list) and raw_positive
            else [row.get("canonical_pair_id")]
        )
        for item_id in positive_ids:
            index = pair_index.get(str(item_id))
            if index is not None:
                positive[query_index, index] = True
        raw_ignored = row.get("ignored_pair_ids", [])
        if not isinstance(raw_ignored, list):
            raise TypeError("ignored_pair_ids must be a list when present")
        for item_id in raw_ignored:
            index = pair_index.get(str(item_id))
            if index is not None:
                ignored[query_index, index] = True
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("every query needs a positive item in the common gallery")
    if torch.any(positive & ignored):
        raise ValueError("positive and ignored masks overlap")
    if torch.any((~ignored).sum(dim=1) == 0):
        raise ValueError("every query needs one valid gallery candidate")
    return positive, ignored


def global_stage_scores(
    text_embeddings: Tensor,
    pair_embeddings: Tensor,
    temperature: Tensor | float = 1.0,
) -> Tensor:
    """Return the stage-1 ANN score on the same scale as the final score."""

    if text_embeddings.ndim != 2 or pair_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if text_embeddings.shape[1] != pair_embeddings.shape[1]:
        raise ValueError("text and pair embedding dimensions differ")
    text = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
    pair = torch.nn.functional.normalize(pair_embeddings.float(), dim=-1)
    return (text @ pair.transpose(0, 1)) / torch.as_tensor(
        temperature, dtype=torch.float32, device=text.device
    )


def merge_reranked_scores(
    global_scores: Tensor, candidate_indices: Tensor, rerank_scores: Tensor
) -> Tensor:
    """Replace only selected candidates while preserving the global gallery."""

    if global_scores.ndim != 2:
        raise ValueError("global_scores must be [Q,P]")
    if candidate_indices.shape != rerank_scores.shape:
        raise ValueError("candidate and rerank shapes differ")
    if candidate_indices.ndim != 2:
        raise ValueError("candidate indices must be [Q,K]")
    if torch.any(candidate_indices < 0) or torch.any(
        candidate_indices >= global_scores.shape[1]
    ):
        raise ValueError("candidate index outside gallery")
    merged = global_scores.clone()
    merged.scatter_(1, candidate_indices, rerank_scores)
    return merged


def ranking_records(
    scores: Tensor,
    query_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Serialize deterministic per-query rankings and exact ranks."""

    if scores.shape != (len(query_rows), len(pair_rows)):
        raise ValueError("score matrix does not align with query/gallery rows")
    k = min(max(int(top_k), 1), scores.shape[1])
    order = scores.argsort(dim=1, descending=True)
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    output: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_rows):
        ranked = order[query_index]
        positives = query.get("positive_pair_ids")
        positive_ids = (
            [str(item) for item in positives]
            if isinstance(positives, list) and positives
            else [str(query["canonical_pair_id"])]
        )
        positive_indices = [
            position + 1
            for position, candidate_index in enumerate(ranked.tolist())
            if pair_ids[candidate_index] in positive_ids
        ]
        output.append(
            {
                "query_id": str(query.get("caption_id", query.get("query_id"))),
                "canonical_pair_id": str(query.get("canonical_pair_id")),
                "positive_pair_ids": positive_ids,
                "rank": min(positive_indices, default=len(pair_ids) + 1),
                "top_pair_ids": [pair_ids[index] for index in ranked[:k].tolist()],
                "top_scores": [float(scores[query_index, index]) for index in ranked[:k]],
            }
        )
    return output


def metrics_by_query_group(
    scores: Tensor,
    relevance: Tensor,
    query_rows: Sequence[dict[str, Any]],
    *,
    ks: Iterable[int] = (1, 5, 10, 50, 100, 500),
    field: str = "dataset_name",
) -> dict[str, dict[str, float]]:
    """Compute the same exact-gallery metrics for source/query subsets."""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(query_rows):
        value = str(row.get(field, "unknown"))
        groups.setdefault(value, []).append(index)
    return {
        group: full_gallery_metrics(scores[indices], relevance[indices], ks=ks)
        for group, indices in sorted(groups.items())
    }


def audit_ranking_integrity(
    scores: Tensor,
    query_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return hashes and alignment facts required for immutable reports."""

    if scores.shape != (len(query_rows), len(pair_rows)):
        raise ValueError("score matrix shape does not match query/gallery lengths")
    query_ids = [str(row.get("caption_id", row.get("query_id"))) for row in query_rows]
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    return {
        "query_count": len(query_ids),
        "gallery_count": len(pair_ids),
        "score_shape": [len(query_ids), len(pair_ids)],
        "query_ids_sha256": _sequence_sha256(query_ids),
        "gallery_ids_sha256": _sequence_sha256(pair_ids),
        "query_ids_unique": len(set(query_ids)) == len(query_ids),
        "gallery_ids_unique": len(set(pair_ids)) == len(pair_ids),
        "scores_finite": bool(torch.isfinite(scores).all()),
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


__all__ = [
    "audit_ranking_integrity",
    "canonical_pair_rows",
    "exact_relevance_masks",
    "global_stage_scores",
    "merge_reranked_scores",
    "metrics_by_query_group",
    "ranking_records",
    "select_topk_candidates",
    "write_jsonl",
]
