"""Deterministic evidence and gates for controlled QCPR v3 experiments.

The three margin families deliberately remain separate: exact instance ranking,
broad teacher semantics, and parser-derived structured near misses.  Only a
human-verified structured benchmark may satisfy the primary grounding gate.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from land_change_detection.models.qcpr import structured_hard_negative_masks
from land_change_detection.models.qcpr_v3 import stable_global_top_n
from land_change_detection.models.qcpr_v3_evaluation import soft_segmentation_metrics, two_stage_retrieval_metrics


def _summary(values: Tensor, *, positive_sizes: Tensor | None = None) -> dict[str, float | int]:
    if values.numel() == 0:
        result: dict[str, float | int] = {"mean": 0.0, "median": 0.0, "p10": 0.0, "minimum": 0.0, "fraction_le_zero": 0.0, "query_count": 0}
    else:
        result = {"mean": float(values.mean()), "median": float(values.median()), "p10": float(torch.quantile(values, 0.1)), "minimum": float(values.min()), "fraction_le_zero": float((values <= 0).float().mean()), "query_count": int(values.numel())}
    if positive_sizes is not None:
        result |= {"positive_set_size_mean": float(positive_sizes.float().mean()) if positive_sizes.numel() else 0.0, "positive_set_size_median": float(positive_sizes.float().median()) if positive_sizes.numel() else 0.0, "positive_set_size_min": int(positive_sizes.min()) if positive_sizes.numel() else 0, "positive_set_size_max": int(positive_sizes.max()) if positive_sizes.numel() else 0}
    return result


def _margin_distribution(scores: Tensor, positive_mask: Tensor, negative_mask: Tensor) -> dict[str, float | int]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape or negative_mask.shape != scores.shape:
        raise ValueError("scores, positive_mask, and negative_mask must have shape [Q,C]")
    positive_mask, negative_mask = positive_mask.bool(), negative_mask.bool() & ~positive_mask.bool()
    valid = positive_mask.any(1) & negative_mask.any(1)
    margin = scores.masked_fill(~positive_mask, float("-inf")).amax(1) - scores.masked_fill(~negative_mask, float("-inf")).amax(1)
    return _summary(margin[valid], positive_sizes=positive_mask.sum(1)[valid])


def exact_pair_positive_nonpair_margin(scores: Tensor, mapping: Tensor) -> dict[str, float | int]:
    if scores.ndim != 2 or mapping.ndim != 1 or scores.shape[0] != mapping.numel():
        raise ValueError("scores [Q,C] and mapping [Q] must align")
    if mapping.numel() and (int(mapping.min()) < 0 or int(mapping.max()) >= scores.shape[1]):
        raise ValueError("mapping contains an out-of-range candidate")
    positives = torch.nn.functional.one_hot(mapping.long(), scores.shape[1]).bool()
    return _margin_distribution(scores, positives, ~positives)


def broad_semantic_margin(scores: Tensor, semantic_relevance: Tensor) -> dict[str, float | int]:
    """Best broad-semantic positive versus strongest broad-semantic negative."""
    return _margin_distribution(scores, semantic_relevance.bool(), ~semantic_relevance.bool())


def parser_derived_near_miss_masks(
    captions: list[str], mapping: Tensor, global_scores: Tensor, broad_positive_mask: Tensor, *, top_n: int
) -> dict[str, Tensor]:
    """Parser-derived, top-N near misses; never call these human verified labels."""
    if global_scores.shape != broad_positive_mask.shape or global_scores.shape[0] != len(captions):
        raise ValueError("captions, global_scores, and broad_positive_mask must align")
    categories = structured_hard_negative_masks(captions, mapping, global_scores.shape[1])
    selected = torch.zeros_like(global_scores, dtype=torch.bool)
    selected.scatter_(1, stable_global_top_n(global_scores, top_n), True)
    names = {
        "same_broad_change_wrong_object": "wrong_object",
        "same_object_wrong_direction": "wrong_direction",
        "same_object_direction_wrong_location": "wrong_location",
        "same_object_location_wrong_count": "wrong_count",
        "no_change_lookalike": "change_no_change_lookalike",
    }
    return {names[name]: mask.to(global_scores.device) & selected & ~broad_positive_mask.bool() for name, mask in categories.items()}


def parser_derived_structured_near_miss_margin(
    scores: Tensor, mapping: Tensor, captions: list[str], global_scores: Tensor, broad_positive_mask: Tensor, *, top_n: int
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    positives = torch.nn.functional.one_hot(mapping.long(), scores.shape[1]).bool()
    categories = parser_derived_near_miss_masks(captions, mapping, global_scores, broad_positive_mask, top_n=top_n)
    union = torch.stack(tuple(categories.values())).any(0) if categories else torch.zeros_like(positives)
    overall = _margin_distribution(scores, positives, union)
    overall["label_provenance"] = "parser_derived"  # type: ignore[assignment]
    overall["status"] = "NOT_HUMAN_VERIFIED"  # type: ignore[assignment]
    by_category = {name: _margin_distribution(scores, positives, mask) for name, mask in categories.items()}
    return overall, by_category


def _prefixed(metrics: Mapping[str, float | int], prefix: str) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def experiment_metrics(
    global_scores: Tensor,
    local_scores: Tensor,
    reranked_scores: Tensor,
    semantic_relevance: Tensor,
    exact_mapping: Tensor,
    *,
    top_n: int,
    captions: list[str] | None = None,
    structured_labels_human_verified: bool = False,
    mask_logits: Tensor | None = None,
    mask_targets: Tensor | None = None,
) -> dict[str, float | int | str | bool]:
    """Canonical metrics. Structured parser labels are evidence, never a PASS gate."""
    metrics: dict[str, float | int | str | bool] = two_stage_retrieval_metrics(global_scores, reranked_scores, semantic_relevance, top_n=top_n)
    metrics.update(_prefixed(exact_pair_positive_nonpair_margin(local_scores, exact_mapping), "local_exact_pair_vs_all_nonpair_margin"))
    metrics.update(_prefixed(exact_pair_positive_nonpair_margin(reranked_scores, exact_mapping), "reranked_exact_pair_vs_all_nonpair_margin"))
    metrics.update(_prefixed(broad_semantic_margin(local_scores, semantic_relevance), "local_broad_semantic_margin"))
    metrics.update(_prefixed(broad_semantic_margin(reranked_scores, semantic_relevance), "reranked_broad_semantic_margin"))
    if captions is None:
        metrics["structured_near_miss_status"] = "NOT_EVALUATED"
    else:
        for score, prefix in ((local_scores, "local_structured_near_miss_margin"), (reranked_scores, "reranked_structured_near_miss_margin")):
            overall, by_category = parser_derived_structured_near_miss_margin(score, exact_mapping, captions, global_scores, semantic_relevance, top_n=top_n)
            metrics.update(_prefixed(overall, prefix))
            for category, values in by_category.items():
                metrics.update(_prefixed(values, f"{prefix}_{category}"))
        metrics["structured_near_miss_status"] = "HUMAN_VERIFIED" if structured_labels_human_verified else "PARSER_DERIVED_NOT_GATE_ELIGIBLE"
    metrics["all_scores_finite"] = bool(torch.isfinite(global_scores).all() and torch.isfinite(local_scores).all() and torch.isfinite(reranked_scores).all())
    if (mask_logits is None) != (mask_targets is None):
        raise ValueError("mask logits and targets must be supplied together")
    if mask_logits is not None and mask_targets is not None:
        metrics.update(soft_segmentation_metrics(mask_logits, mask_targets))
    return metrics



def margin_family_reports(
    local_scores: Tensor,
    reranked_scores: Tensor,
    exact_mapping: Tensor,
    semantic_relevance: Tensor,
    captions: list[str],
    global_scores: Tensor,
    *,
    top_n: int,
    query_groups: Mapping[str, Tensor] | None = None,
) -> dict[str, dict[str, object]]:
    """Nested audit; parser-derived structured labels are not gate eligible."""
    groups: dict[str, Tensor] = {"all": torch.ones(exact_mapping.numel(), dtype=torch.bool, device=exact_mapping.device)}
    if query_groups:
        for name, keep in query_groups.items():
            if keep.shape != exact_mapping.shape:
                raise ValueError(f"query group {name!r} must have shape [Q]")
            groups[name] = keep.to(device=exact_mapping.device, dtype=torch.bool)
    result: dict[str, dict[str, object]] = {}
    for group, keep in groups.items():
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        if not indices.numel():
            result[group] = {"query_count": 0, "status": "NOT_EVALUATED"}
            continue
        mapping = exact_mapping[indices]
        local, reranked = local_scores[indices], reranked_scores[indices]
        semantic, global_ = semantic_relevance[indices], global_scores[indices]
        subset_captions = [captions[index] for index in indices.tolist()]
        local_structured, local_categories = parser_derived_structured_near_miss_margin(local, mapping, subset_captions, global_, semantic, top_n=top_n)
        reranked_structured, reranked_categories = parser_derived_structured_near_miss_margin(reranked, mapping, subset_captions, global_, semantic, top_n=top_n)
        result[group] = {
            "query_count": int(indices.numel()),
            "local_exact_pair_vs_all_nonpair_margin": exact_pair_positive_nonpair_margin(local, mapping),
            "reranked_exact_pair_vs_all_nonpair_margin": exact_pair_positive_nonpair_margin(reranked, mapping),
            "local_broad_semantic_margin": broad_semantic_margin(local, semantic),
            "reranked_broad_semantic_margin": broad_semantic_margin(reranked, semantic),
            "local_structured_near_miss_margin": local_structured,
            "reranked_structured_near_miss_margin": reranked_structured,
            "local_structured_near_miss_by_category": local_categories,
            "reranked_structured_near_miss_by_category": reranked_categories,
            "structured_label_provenance": "parser_derived_not_gate_eligible",
        }
    return result

def acceptance_gates(phase: str, metrics: Mapping[str, float | int | str | bool], *, v1_global_r1: float, v1_global_r5: float) -> dict[str, bool]:
    if phase not in {"late_interaction", "mask_grounding"}:
        raise ValueError(f"controlled gates are defined only for B/C, got {phase!r}")
    global_r1, global_r5 = float(metrics["global_semantic_r1"]), float(metrics["global_semantic_r5"])
    rerank_r1, rerank_r5 = float(metrics["reranked_semantic_r1"]), float(metrics["reranked_semantic_r5"])
    structured_status = str(metrics.get("structured_near_miss_status", "NOT_EVALUATED"))
    structured_mean = float(metrics.get("local_structured_near_miss_margin_mean", 0.0))
    structured_median = float(metrics.get("local_structured_near_miss_margin_median", 0.0))
    gates: dict[str, bool] = {
        "global_r1_within_0.01_of_v1": global_r1 >= v1_global_r1 - 0.01,
        "global_r5_within_0.01_of_v1": global_r5 >= v1_global_r5 - 0.01,
        "candidate_recall_reported": "candidate_recall_at_n" in metrics,
        "conditional_reranking_gain_positive": float(metrics["conditional_reranking_gain_r1"]) > 0.0,
        "structured_near_miss_human_verified": structured_status == "HUMAN_VERIFIED",
        "structured_near_miss_mean_positive": structured_mean > 0.0,
        "structured_near_miss_median_positive": structured_median > 0.0,
        "reranked_r1_not_worse_than_global": rerank_r1 >= global_r1,
        "reranked_r5_not_worse_than_global": rerank_r5 >= global_r5,
        "all_values_finite": bool(metrics["all_scores_finite"]),
        "local_gradients_finite_nonzero": bool(metrics.get("local_gradients_finite_nonzero", False)),
    }
    if phase == "mask_grounding":
        gates.update(mask_not_collapsed=float(metrics["predicted_area"]) > 0.0, nonempty_recall_above_zero=float(metrics["recall"]) > 0.0, empty_false_positive_reported="empty_false_positive_area" in metrics, faithful_mask=bool(metrics.get("faithful_mask", False)))
    return gates


def parser_derived_late_interaction_loss(
    token_patch_scores: Tensor,
    mapping: Tensor,
    captions: list[str],
    global_scores: Tensor,
    broad_positive_mask: Tensor,
    *,
    top_n: int,
    margin: float = 0.1,
) -> tuple[Tensor, dict[str, float | int]]:
    """Top-N generic late-interaction loss against filtered parser near misses.

    Broad semantic positives are excluded before mining. A query with no
    trustworthy parser-derived conflict contributes an exact zero-gradient loss.
    """
    categories = parser_derived_near_miss_masks(captions, mapping, global_scores, broad_positive_mask, top_n=top_n)
    hard = torch.stack(tuple(categories.values())).any(0) if categories else torch.zeros_like(token_patch_scores, dtype=torch.bool)
    rows = torch.arange(mapping.numel(), device=token_patch_scores.device)
    positive = token_patch_scores[rows, mapping.long()]
    negative = token_patch_scores.masked_fill(~hard, float("-inf")).amax(1)
    valid = torch.isfinite(negative)
    loss = torch.relu(float(margin) - positive[valid] + negative[valid]).mean() if bool(valid.any()) else token_patch_scores.sum() * 0.0
    result: dict[str, float | int] = {
        "structured_training_valid_queries": int(valid.sum()),
        "structured_training_positive_mean": float(positive[valid].mean()) if bool(valid.any()) else 0.0,
        "structured_training_positive_median": float(positive[valid].median()) if bool(valid.any()) else 0.0,
        "structured_training_hard_negative_mean": float(negative[valid].mean()) if bool(valid.any()) else 0.0,
        "structured_training_hard_negative_median": float(negative[valid].median()) if bool(valid.any()) else 0.0,
        "structured_training_margin_mean": float((positive[valid] - negative[valid]).mean()) if bool(valid.any()) else 0.0,
        "structured_training_loss": float(loss.detach()),
    }
    for name, category in categories.items():
        result[f"structured_training_{name}_count"] = int(category.sum())
    return loss, result
