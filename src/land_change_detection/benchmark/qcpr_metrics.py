"""Protocol-explicit retrieval metrics for the QCPR benchmark.

The functions operate on ranked physical IDs and relevance sets; no model or
dataset-specific assumptions are hidden in the implementation.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import math


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    for index, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary query-level success: at least one relevant item in Top-K."""
    return float(bool(set(ranking[:k]) & relevant)) if relevant else 0.0


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items recovered in Top-K."""
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of Top-K slots that are relevant."""
    return len(set(ranking[:k]) & relevant) / k if k > 0 else 0.0


def mrr_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item, truncated to Top-K."""
    for index, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def recall_at(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Backward-compatible alias for fraction-of-relevant Recall@K."""
    return recall_at_k(ranking, relevant, k)


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
        "hit@1": sum(hit_rate_at_k(r, rel, 1) for r, rel in values) / len(values) if values else 0.0,
        "hit@5": sum(hit_rate_at_k(r, rel, 5) for r, rel in values) / len(values) if values else 0.0,
        "hit@10": sum(hit_rate_at_k(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
        "recall@1": sum(recall_at_k(r, rel, 1) for r, rel in values) / len(values) if values else 0.0,
        "recall@5": sum(recall_at_k(r, rel, 5) for r, rel in values) / len(values) if values else 0.0,
        "recall@10": sum(recall_at_k(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
        "recall@50": sum(recall_at_k(r, rel, 50) for r, rel in values) / len(values) if values else 0.0,
        "recall@100": sum(recall_at_k(r, rel, 100) for r, rel in values) / len(values) if values else 0.0,
        "ndcg@10": sum(ndcg_at(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
        "mean_rank": sum((next((i for i, x in enumerate(r, 1) if x in rel), len(r) + 1) for r, rel in values), 0) / len(values) if values else 0.0,
        "median_rank": float(__import__("statistics").median([next((i for i, x in enumerate(r, 1) if x in rel), len(r) + 1) for r, rel in values])) if values else 0.0,
    }


def average_precision(ranking: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, item in enumerate(ranking, 1):
        if item in relevant:
            hits += 1
            total += hits / index
    return total / len(relevant)


def pair_to_text_metrics(rankings: Iterable[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    values = list(rankings)
    return {
        "queries": len(values),
        "mAP": sum(average_precision(r, rel) for r, rel in values) / len(values) if values else 0.0,
        "MRR": sum(reciprocal_rank(r, rel) for r, rel in values) / len(values) if values else 0.0,
        "R@1": sum(recall_at_k(r, rel, 1) for r, rel in values) / len(values) if values else 0.0,
        "R@5": sum(recall_at_k(r, rel, 5) for r, rel in values) / len(values) if values else 0.0,
        "R@10": sum(recall_at_k(r, rel, 10) for r, rel in values) / len(values) if values else 0.0,
    }


def text_overlap(candidate: str, references: Iterable[str]) -> dict[str, float]:
    """Protocol-neutral token overlap diagnostic, not a BLEU replacement."""
    cand = set(str(candidate).casefold().split())
    ref = set(token for text in references for token in str(text).casefold().split())
    precision = len(cand & ref) / len(cand) if cand else 0.0
    recall = len(cand & ref) / len(ref) if ref else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def energy_inside_mask(probabilities: Sequence[float], mask: Sequence[bool]) -> float:
    total = sum(float(x) for x in probabilities)
    return sum(float(x) for x, m in zip(probabilities, mask) if m) / total if total else 0.0


def soft_iou(probabilities: Sequence[float], mask: Sequence[bool], threshold: float = 0.5) -> float:
    pred = [float(x) >= threshold for x in probabilities]
    inter = sum(p and m for p, m in zip(pred, mask)); union = sum(p or m for p, m in zip(pred, mask))
    return inter / union if union else 1.0


def auprc(probabilities: Sequence[float], mask: Sequence[bool]) -> float:
    order = sorted(range(len(probabilities)), key=lambda i: float(probabilities[i]), reverse=True)
    positives = sum(bool(x) for x in mask)
    if not positives: return 0.0
    hits = 0; area = 0.0
    for rank, index in enumerate(order, 1):
        if mask[index]:
            hits += 1; area += hits / rank
    return area / positives


def pointing_game(probabilities: Sequence[float], mask: Sequence[bool]) -> float:
    return float(bool(mask[max(range(len(probabilities)), key=lambda i: float(probabilities[i]))])) if probabilities else 0.0


def normalized_entropy(probabilities: Sequence[float]) -> float:
    values = [max(float(x), 0.0) for x in probabilities]; total = sum(values)
    if not values or not total: return 0.0
    return -sum((x / total) * math.log(x / total) for x in values if x) / math.log(len(values)) if len(values) > 1 else 0.0


def effective_patch_count(probabilities: Sequence[float]) -> float:
    values = [max(float(x), 0.0) for x in probabilities]; total = sum(values)
    return (total * total / sum(x * x for x in values)) if total and any(values) else 0.0


def deletion_insertion_auc(scores: Sequence[float], mask: Sequence[bool]) -> dict[str, float]:
    """A simple deterministic evidence curve over cumulative ranked tokens."""
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    total = sum(float(x) for x in scores) or 1.0
    curve = [sum(float(scores[i]) for i in order[:k]) / total for k in range(1, len(order) + 1)]
    return {"deletion_auc": 1.0 - sum(curve) / len(curve) if curve else 0.0, "insertion_auc": sum(curve) / len(curve) if curve else 0.0}


def query_swap_sensitivity(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or not second: return 0.0
    return sum(abs(float(a) - float(b)) for a, b in zip(first, second)) / min(len(first), len(second))


def QCPR_EXACT_FULL_GALLERY(rows: Iterable[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    return aggregate_rankings(rows)


def QCPR_SEMANTIC_MULTI_RELEVANCE(rows: Iterable[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    return aggregate_rankings(rows)


def CHANGERETCAP_FNA_TOP5(ranking: Sequence[str], relevant: set[str]) -> dict[str, float]:
    return changeretcap_compat(ranking, relevant, k=5)


def TEXT_ITSR_CAPTION_OVERLAP(candidate: str, references: Iterable[str]) -> dict[str, float]:
    return text_overlap(candidate, references)


def DFM_CAPTIONING(candidate: str, references: Iterable[str]) -> dict[str, float]:
    return text_overlap(candidate, references)


def DGTRS_STATIC_RETRIEVAL(rows: Iterable[tuple[Sequence[str], set[str]]]) -> dict[str, float]:
    return aggregate_rankings(rows)


def AIR_SLT_COMPAT(probabilities: Sequence[float], mask: Sequence[bool]) -> dict[str, float]:
    return {"pointing_game": pointing_game(probabilities, mask), "energy_inside_mask": energy_inside_mask(probabilities, mask), "normalized_entropy": normalized_entropy(probabilities)}


def QCPR_TEMPORAL_SOFT_LOCALIZATION(probabilities: Sequence[float], mask: Sequence[bool]) -> dict[str, float]:
    result = AIR_SLT_COMPAT(probabilities, mask)
    result.update({"soft_iou": soft_iou(probabilities, mask), "auprc": auprc(probabilities, mask), "effective_patch_count": effective_patch_count(probabilities), "deletion_auc": deletion_insertion_auc(probabilities, mask)["deletion_auc"], "insertion_auc": deletion_insertion_auc(probabilities, mask)["insertion_auc"]})
    return result


def changeretcap_compat(ranking: Sequence[str], relevant: set[str], k: int = 5) -> dict[str, float]:
    """Top-k compatibility view; it is not interchangeable with full-gallery MRR."""
    top = list(ranking[:k])
    return {
        "k": k,
        "hit_rate": hit_rate_at_k(ranking, relevant, k),
        "recall": recall_at_k(ranking, relevant, k),
        "precision": precision_at_k(ranking, relevant, k),
        "mrr@k": mrr_at_k(ranking, relevant, k),
    }
