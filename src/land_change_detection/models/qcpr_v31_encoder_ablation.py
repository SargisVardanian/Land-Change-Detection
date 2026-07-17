"""Shared contracts for the QCPR v3.1 frozen-encoder ablation.

The alternative VLMs are single-image encoders.  Their zero-shot temporal
score is therefore reported separately from the learned QCPR pair encoder and
must never be presented as an architecture-compatible checkpoint replacement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def normalized_temporal_delta(before: Tensor, after: Tensor) -> Tensor:
    """Return the signed, normalized after-before feature.

    The subtraction happens before normalization so temporal reversal changes
    the sign exactly (up to floating-point precision).
    """
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("before and after must have the same [B,D] shape")
    return F.normalize(after - before, dim=-1)


def score_zero_shot_temporal_delta(text: Tensor, before: Tensor, after: Tensor) -> Tensor:
    if text.ndim != 2 or text.shape[-1] != before.shape[-1]:
        raise ValueError("text and image embeddings must share their final dimension")
    return F.normalize(text, dim=-1) @ normalized_temporal_delta(before, after).T


def retrieval_metrics(scores: Tensor, relevance: Tensor, exact_pair: Tensor) -> dict[str, Any]:
    """Compute deterministic retrieval metrics for one score matrix."""
    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have matching [Q,C] shapes")
    if exact_pair.shape != (scores.shape[0],):
        raise ValueError("exact_pair must be [Q]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    if relevance.dtype != torch.bool:
        relevance = relevance.bool()
    if (relevance.sum(dim=1) == 0).any():
        raise ValueError("every query must have at least one semantic positive")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranked_relevance = relevance.gather(1, order)
    exact_hits = order.eq(exact_pair[:, None])
    result: dict[str, Any] = {
        "query_count": int(scores.shape[0]),
        "candidate_count": int(scores.shape[1]),
        "positive_set_size": {
            "mean": float(relevance.sum(1).float().mean()),
            "median": float(relevance.sum(1).float().median()),
            "minimum": int(relevance.sum(1).min()),
            "maximum": int(relevance.sum(1).max()),
        },
    }
    for k in (1, 5, 10):
        kk = min(k, scores.shape[1])
        result[f"semantic_r{k}"] = float(ranked_relevance[:, :kk].any(1).float().mean())
        result[f"exact_pair_r{k}_diagnostic"] = float(exact_hits[:, :kk].any(1).float().mean())
    for k in (50, 100, 200):
        kk = min(k, scores.shape[1])
        result[f"semantic_candidate_recall_at_{k}"] = float(
            ranked_relevance[:, :kk].any(1).float().mean()
        )
        result[f"exact_pair_candidate_recall_at_{k}_diagnostic"] = float(
            exact_hits[:, :kk].any(1).float().mean()
        )
    k = min(10, scores.shape[1])
    discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
    dcg = (ranked_relevance[:, :k].float() * discounts).sum(1)
    ideal_counts = relevance.sum(1).clamp(max=k)
    idcg = torch.stack([discounts[: int(count)].sum() for count in ideal_counts])
    result["semantic_ndcg_at_10"] = float((dcg / idcg.clamp_min(1e-12)).mean())
    return result


def metrics_by_query_groups(
    scores: Tensor,
    relevance: Tensor,
    exact_pair: Tensor,
    groups: dict[str, Tensor],
) -> dict[str, Any]:
    report = {"all": retrieval_metrics(scores, relevance, exact_pair)}
    for name, mask in groups.items():
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.shape != (scores.shape[0],):
            raise ValueError(f"query group {name!r} must be [Q]")
        report[name] = (
            retrieval_metrics(scores[mask], relevance[mask], exact_pair[mask])
            if mask.any()
            else {"query_count": 0, "status": "NOT_EVALUATED"}
        )
    return report


def replacement_gate(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Apply the declared no-regression gate to comparable global metrics."""
    keys = ("semantic_r1", "semantic_r5", "semantic_r10", "semantic_ndcg_at_10")
    checks = {
        key: bool(float(candidate[key]) >= float(anchor[key]) - tolerance)
        for key in keys
    }
    improved = any(float(candidate[key]) > float(anchor[key]) for key in keys)
    return {
        "comparable_contract": True,
        "tolerance": tolerance,
        "checks": checks,
        "at_least_one_primary_metric_improved": improved,
        "pass": bool(all(checks.values()) and improved),
    }
