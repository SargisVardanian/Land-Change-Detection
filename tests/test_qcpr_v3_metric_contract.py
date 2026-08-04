from __future__ import annotations

import torch

from qcpr_v3.evaluation.retrieval import (
    hit_rate_at_k,
    mean_average_precision,
    mrr_at_k,
    pair_to_text_metrics,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
)


def test_hit_and_recall_differ_for_multi_positive_relevance():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    relevant = torch.tensor([[True, False, False, True]])
    assert hit_rate_at_k(scores, relevant, 2) == 1.0
    assert recall_at_k(scores, relevant, 2) == 0.5
    assert precision_at_k(scores, relevant, 2) == 0.5


def test_truncated_mrr_differs_from_full_gallery_mrr():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    relevant = torch.tensor([[False, False, False, True]])
    assert mrr_at_k(scores, relevant, 3) == 0.0
    assert retrieval_metrics(scores, relevant, ks=(3,))["mrr"] == 0.25


def test_map_and_reverse_direction_use_all_valid_positives():
    scores = torch.tensor([[4.0, 3.0, 2.0], [3.0, 1.0, 2.0]])
    relevant = torch.tensor([[True, False, True], [False, True, False]])
    assert abs(mean_average_precision(scores, relevant) - ((5.0 / 6.0) + (1.0 / 3.0)) / 2.0) < 1e-6
    reverse = pair_to_text_metrics(scores, relevant, ks=(1,))
    assert reverse["mrr"] > 0.0
    assert "ndcg_at_1" in retrieval_metrics(scores, relevant, ks=(1,))
