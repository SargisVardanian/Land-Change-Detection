from __future__ import annotations

import torch
from torch import Tensor

from land_change_detection.models.qcpr_v3 import apply_two_stage_reranking, candidate_recall_at_n


def _retrieval_summary(scores: Tensor, relevance: Tensor, prefix: str) -> dict[str, float | int]:
    relevant = relevance.bool()
    valid = relevant.any(dim=1)
    scores = scores[valid]; relevant = relevant[valid]
    if not scores.numel():
        return {f"{prefix}_queries": 0, **{f"{prefix}_r{k}": 0.0 for k in (1, 5, 10)}, f"{prefix}_ndcg10": 0.0}
    order = scores.argsort(dim=1, descending=True, stable=True)
    ranked = relevant.gather(1, order)
    result: dict[str, float | int] = {f"{prefix}_queries": int(valid.sum())}
    for k in (1, 5, 10):
        result[f"{prefix}_r{k}"] = float(ranked[:, : min(k, ranked.shape[1])].any(dim=1).float().mean())
    gains = ranked[:, : min(10, ranked.shape[1])].float()
    discounts = 1 / torch.log2(torch.arange(gains.shape[1], device=scores.device).float() + 2)
    dcg = (gains * discounts).sum(1)
    ideal_count = relevant.sum(1).clamp_max(gains.shape[1])
    idcg = torch.stack([discounts[: int(count)].sum() for count in ideal_count]).clamp_min(1e-8)
    result[f"{prefix}_ndcg10"] = float((dcg / idcg).mean())
    return result


def two_stage_retrieval_metrics(global_scores: Tensor, reranked_scores: Tensor, semantic_relevance: Tensor, *, top_n: int) -> dict[str, float | int]:
    staged = apply_two_stage_reranking(global_scores, reranked_scores, top_n)
    global_metrics = _retrieval_summary(global_scores, semantic_relevance, "global_semantic")
    rerank_metrics = _retrieval_summary(staged, semantic_relevance, "reranked_semantic")
    result = global_metrics | rerank_metrics
    result["candidate_recall_at_n"] = float(candidate_recall_at_n(global_scores, semantic_relevance, top_n))
    for k in (1, 5, 10):
        result[f"conditional_reranking_gain_r{k}"] = float(result[f"reranked_semantic_r{k}"] - result[f"global_semantic_r{k}"])
    global_top = global_scores.argsort(dim=1, descending=True, stable=True)
    positive = semantic_relevance.bool()
    result["positives_lost_before_reranking"] = int((~positive.gather(1, global_top[:, : min(top_n, global_top.shape[1])]).any(1)).sum())
    return result


def soft_segmentation_metrics(logits: Tensor, targets: Tensor, *, threshold: float = 0.5) -> dict[str, float | int]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    probabilities = logits.sigmoid().flatten(1)
    truth = targets.bool().flatten(1)
    predicted = probabilities >= threshold
    nonempty = truth.any(1); empty = ~nonempty
    tp = (predicted & truth).sum(1).float(); fp = (predicted & ~truth).sum(1).float(); fn = (~predicted & truth).sum(1).float()
    def selected_mean(value: Tensor, select: Tensor) -> float:
        return float(value[select].mean()) if bool(select.any()) else 0.0
    max_index = probabilities.argmax(1)
    pointing = truth.gather(1, max_index[:, None]).squeeze(1).float()
    return {
        "nonempty_count": int(nonempty.sum()), "empty_count": int(empty.sum()),
        "nonempty_dice": selected_mean((2 * tp + 1) / (2 * tp + fp + fn + 1), nonempty),
        "nonempty_iou": selected_mean((tp + 1) / (tp + fp + fn + 1), nonempty),
        "precision": selected_mean((tp + 1) / (tp + fp + 1), nonempty),
        "recall": selected_mean((tp + 1) / (tp + fn + 1), nonempty),
        "pointing_game": selected_mean(pointing, nonempty),
        "empty_false_positive_area": selected_mean(predicted.float().mean(1), empty),
        "predicted_area": float(predicted.float().mean()), "target_area": float(truth.float().mean()),
    }
