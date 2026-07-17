from __future__ import annotations

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from land_change_detection.models.qcpr_v31_encoder_ablation import (
    load_global_retrieval_modules_strict,
    metrics_by_query_groups,
    normalized_temporal_delta,
    pooled_feature_tensor,
    replacement_gate,
    retrieval_metrics,
    score_zero_shot_temporal_delta,
    stable_fingerprint,
)


class _GlobalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = nn.Linear(2, 2)
        self.temporal_encoder = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.retrieval_head = nn.Linear(2, 2)
        self.text_adapter = nn.Linear(2, 2)
        self.grounder = nn.Linear(3, 3)


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


def test_global_checkpoint_load_is_strict_but_excludes_versioned_grounder() -> None:
    source = _GlobalModel()
    target = _GlobalModel()
    old_state = source.state_dict()
    # Simulate a checkpoint whose grounder architecture differs.
    old_state = {
        key: value
        for key, value in old_state.items()
        if not key.startswith("grounder.")
    } | {"grounder.old_decoder.weight": torch.ones(1)}
    report = load_global_retrieval_modules_strict(target, old_state)
    assert report["global_path_strict"] is True
    assert report["excluded_prefixes"] == {"grounder": 1}
    for name in (
        "visual_encoder",
        "temporal_encoder",
        "text_encoder",
        "retrieval_head",
        "text_adapter",
    ):
        torch.testing.assert_close(
            getattr(source, name).weight,
            getattr(target, name).weight,
        )


def test_global_checkpoint_load_rejects_missing_global_key() -> None:
    model = _GlobalModel()
    state = model.state_dict()
    state.pop("temporal_encoder.weight")
    with pytest.raises(RuntimeError, match="temporal_encoder"):
        load_global_retrieval_modules_strict(_GlobalModel(), state)


def test_global_checkpoint_load_rejects_unknown_non_grounder_prefix() -> None:
    model = _GlobalModel()
    state = model.state_dict() | {"mystery.weight": torch.ones(1)}
    with pytest.raises(RuntimeError, match="unrecognized"):
        load_global_retrieval_modules_strict(_GlobalModel(), state)


@pytest.mark.parametrize(
    "output",
    (
        torch.ones(2, 3),
        SimpleNamespace(pooler_output=torch.ones(2, 3)),
        (torch.ones(2, 4, 3), torch.ones(2, 3)),
    ),
)
def test_pooled_feature_tensor_supports_transformers_return_contracts(output) -> None:
    feature = pooled_feature_tensor(output, expected_batch=2)
    assert feature.shape == (2, 3)
    assert torch.isfinite(feature).all()


def test_pooled_feature_tensor_rejects_wrong_batch_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="B=2"):
        pooled_feature_tensor(torch.ones(1, 3), expected_batch=2)
    with pytest.raises(ValueError, match="NaN"):
        pooled_feature_tensor(torch.tensor([[float("nan")]]), expected_batch=1)
