"""Deterministic experiment metrics and preregistered acceptance gates.

This module deliberately contains no neural scoring equations.  The model owns
all scoring; this layer only turns canonical score tensors into the evidence
required to decide whether a controlled B/C diagnostic may advance.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from land_change_detection.models.qcpr_v3_evaluation import (
    soft_segmentation_metrics,
    two_stage_retrieval_metrics,
)


def positive_hard_negative_margin(scores: Tensor, mapping: Tensor) -> float:
    """Exact-pair positive minus strongest non-pair candidate, averaged."""
    if scores.ndim != 2 or mapping.ndim != 1 or scores.shape[0] != mapping.numel():
        raise ValueError("scores [Q,C] and mapping [Q] must align")
    if mapping.numel() == 0:
        return 0.0
    if int(mapping.min()) < 0 or int(mapping.max()) >= scores.shape[1]:
        raise ValueError("mapping contains an out-of-range candidate")
    rows = torch.arange(mapping.numel(), device=scores.device)
    positive = scores[rows, mapping]
    negative = scores.masked_fill(torch.nn.functional.one_hot(mapping, scores.shape[1]).bool(), float("-inf")).amax(1)
    return float((positive - negative).mean())


def experiment_metrics(
    global_scores: Tensor,
    local_scores: Tensor,
    reranked_scores: Tensor,
    semantic_relevance: Tensor,
    exact_mapping: Tensor,
    *,
    top_n: int,
    mask_logits: Tensor | None = None,
    mask_targets: Tensor | None = None,
) -> dict[str, float | int]:
    """Canonical metrics used for both diagnostics and controlled pilots."""
    metrics = two_stage_retrieval_metrics(global_scores, reranked_scores, semantic_relevance, top_n=top_n)
    metrics["local_positive_hard_negative_margin"] = positive_hard_negative_margin(local_scores, exact_mapping)
    metrics["reranked_positive_hard_negative_margin"] = positive_hard_negative_margin(reranked_scores, exact_mapping)
    metrics["all_scores_finite"] = bool(
        torch.isfinite(global_scores).all()
        and torch.isfinite(local_scores).all()
        and torch.isfinite(reranked_scores).all()
    )
    if (mask_logits is None) != (mask_targets is None):
        raise ValueError("mask logits and targets must be supplied together")
    if mask_logits is not None and mask_targets is not None:
        metrics.update(soft_segmentation_metrics(mask_logits, mask_targets))
    return metrics


def acceptance_gates(
    phase: str,
    metrics: Mapping[str, float | int | bool],
    *,
    v1_global_r1: float,
    v1_global_r5: float,
) -> dict[str, bool]:
    """Apply the fixed B/C preregistered gates without silently relaxing any."""
    if phase not in {"late_interaction", "mask_grounding"}:
        raise ValueError(f"controlled gates are defined only for B/C, got {phase!r}")
    global_r1 = float(metrics["global_semantic_r1"])
    global_r5 = float(metrics["global_semantic_r5"])
    rerank_r1 = float(metrics["reranked_semantic_r1"])
    rerank_r5 = float(metrics["reranked_semantic_r5"])
    gates: dict[str, bool] = {
        "global_r1_within_0.01_of_v1": global_r1 >= v1_global_r1 - 0.01,
        "global_r5_within_0.01_of_v1": global_r5 >= v1_global_r5 - 0.01,
        "candidate_recall_reported": "candidate_recall_at_n" in metrics,
        "local_positive_hard_negative_margin_positive": float(metrics["local_positive_hard_negative_margin"]) > 0.0,
        "reranked_r1_not_worse_than_global": rerank_r1 >= global_r1,
        "reranked_r5_not_worse_than_global": rerank_r5 >= global_r5,
        "all_values_finite": bool(metrics["all_scores_finite"]),
        "local_gradients_finite_nonzero": bool(metrics.get("local_gradients_finite_nonzero", False)),
    }
    if phase == "mask_grounding":
        predicted = float(metrics["predicted_area"])
        gates.update(
            mask_not_collapsed=predicted > 0.0,
            nonempty_recall_above_zero=float(metrics["recall"]) > 0.0,
            empty_false_positive_reported="empty_false_positive_area" in metrics,
            faithful_mask=bool(metrics.get("faithful_mask", False)),
        )
    return gates
