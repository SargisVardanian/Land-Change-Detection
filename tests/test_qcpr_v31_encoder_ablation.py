from __future__ import annotations

import pytest
import torch

from land_change_detection.models.qcpr_v31_encoder_ablation import (
    metrics_by_query_groups,
    normalized_temporal_delta,
    replacement_gate,
    retrieval_metrics,
    score_zero_shot_temporal_delta,
    stable_fingerprint,
)


def test_temporal_delta_reversal_changes_sign() -> None:
    before = torch.tensor([[1.0, 2.0], [4.0, -1.0]])
    after = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    forward = normalized_temporal_delta(before, after)
    reverse = normalized_temporal_delta(after, before)
    assert torch.allclose(forward, -reverse)


def test_zero_shot_temporal_score_has_expected_shape() -> None:
    text = torch.eye(2)
    before = torch.zeros(3, 2)
    after = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scores = score_zero_shot_temporal_delta(text, before, after)
    assert scores.shape == (2, 3)
    assert scores[0].argmax().item() == 0
    assert scores[1].argmax().item() == 1


def test_retrieval_metrics_are_deterministic_and_exact_pair_is_diagnostic() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.7, 0.6]])
    relevance = torch.tensor([[True, True, False], [False, False, True]])
    mapping = torch.tensor([1, 2])
    report = retrieval_metrics(scores, relevance, mapping)
    assert report["semantic_r1"] == pytest.approx(0.5)
    assert report["semantic_r5"] == pytest.approx(1.0)
    assert report["exact_pair_r1_diagnostic"] == pytest.approx(0.0)
    assert report["semantic_candidate_recall_at_50"] == pytest.approx(1.0)


def test_query_group_metrics_preserve_empty_group_as_not_evaluated() -> None:
    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    relevance = torch.eye(2, dtype=torch.bool)
    mapping = torch.arange(2)
    report = metrics_by_query_groups(
        scores,
        relevance,
        mapping,
        {"first": torch.tensor([True, False]), "empty": torch.zeros(2, dtype=torch.bool)},
    )
    assert report["first"]["semantic_r1"] == pytest.approx(1.0)
    assert report["empty"]["status"] == "NOT_EVALUATED"


def test_replacement_gate_requires_no_regression_and_improvement() -> None:
    anchor = {
        "semantic_r1": 0.5,
        "semantic_r5": 0.7,
        "semantic_r10": 0.8,
        "semantic_ndcg_at_10": 0.6,
    }
    candidate = dict(anchor, semantic_r1=0.51)
    assert replacement_gate(anchor, candidate)["pass"] is True
    candidate["semantic_r5"] = 0.68
    assert replacement_gate(anchor, candidate)["pass"] is False


def test_stable_fingerprint_changes_with_order() -> None:
    assert stable_fingerprint(["a", "b"]) == stable_fingerprint(["a", "b"])
    assert stable_fingerprint(["a", "b"]) != stable_fingerprint(["b", "a"])


def test_invalid_temporal_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same"):
        normalized_temporal_delta(torch.zeros(2, 3), torch.zeros(3, 3))
