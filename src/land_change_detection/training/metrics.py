from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(relevances: Sequence[bool], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevances:
        return 0.0
    return 1.0 if any(relevances[:k]) else 0.0


def mean_average_precision(relevance_lists: Sequence[Sequence[bool]]) -> float:
    if not relevance_lists:
        return 0.0
    ap_values: list[float] = []
    for relevances in relevance_lists:
        hits = 0
        precision_sum = 0.0
        for index, relevant in enumerate(relevances, start=1):
            if not relevant:
                continue
            hits += 1
            precision_sum += hits / index
        ap_values.append(precision_sum / hits if hits else 0.0)
    return sum(ap_values) / len(ap_values)


def transition_consistency_score(expected: str | None, predicted: Sequence[str]) -> float:
    if expected is None:
        return 0.0
    if not predicted:
        return 0.0
    return 1.0 if expected in predicted else 0.0
