from __future__ import annotations

import json

import pytest
import torch

from qcpr_v3.evaluation.retrieval import (
    hit_rate_at_k,
    mrr_at_k,
    mrr_full,
    recall_at_k,
    retrieval_metrics,
)


def test_multi_positive_hit_and_recall_are_distinct() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    relevant = torch.tensor([[True, True, False, False]])
    assert hit_rate_at_k(scores, relevant, 1) == 1.0
    assert recall_at_k(scores, relevant, 1) == 0.5
    assert hit_rate_at_k(scores, relevant, 1) != recall_at_k(scores, relevant, 1)


def test_retrieval_metrics_use_explicit_public_names() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    relevant = torch.tensor([[True, True, False, False]])
    result = retrieval_metrics(scores, relevant, ks=(1, 4))
    serialized = json.dumps(dict(result), sort_keys=True)
    assert "candidate_hit_at_1" in result
    assert "multi_positive_recall_at_1" in result
    assert "candidate_recall" not in result
    assert "candidate_recall" not in serialized


def test_truncated_mrr_differs_from_full_gallery_mrr() -> None:
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    relevant = torch.tensor([[False, False, True, False]])
    assert mrr_at_k(scores, relevant, 1) == 0.0
    assert mrr_full(scores, relevant) == pytest.approx(1.0 / 3.0)
