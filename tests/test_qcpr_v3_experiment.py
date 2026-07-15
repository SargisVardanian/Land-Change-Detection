from __future__ import annotations

import pytest
import torch

from land_change_detection.models.qcpr_v3_experiment import acceptance_gates, broad_semantic_margin, experiment_metrics, margin_family_reports


def _scores():
    global_scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    local_scores = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    reranked_scores = global_scores + local_scores
    relevance = torch.tensor([[True, False, False], [False, True, False]])
    mapping = torch.tensor([0, 1])
    return global_scores, local_scores, reranked_scores, relevance, mapping


def test_broad_margin_excludes_relevant_duplicate_from_negative_pool() -> None:
    scores = torch.tensor([[0.4, 0.9, 0.1]])
    relevance = torch.tensor([[True, True, False]])
    assert broad_semantic_margin(scores, relevance)["mean"] == pytest.approx(0.8)
    metrics = experiment_metrics(scores, scores, scores, relevance, torch.tensor([0]), top_n=2)
    assert metrics["local_broad_semantic_margin_mean"] == pytest.approx(0.8)
    assert metrics["local_exact_pair_vs_all_nonpair_margin_mean"] == pytest.approx(-0.5)
    assert metrics["structured_near_miss_status"] == "NOT_EVALUATED"


def test_parser_derived_near_miss_is_not_b_gate_eligible() -> None:
    scores = torch.tensor([[0.8, 0.7, 0.1], [0.7, 0.8, 0.1], [0.1, 0.2, 0.8]])
    relevance = torch.eye(3, dtype=torch.bool)
    metrics = experiment_metrics(scores, scores, scores, relevance, torch.tensor([0, 1, 2]), top_n=3, captions=["a house appeared on the left", "a house appeared on the right", "a road appeared on the left"])
    assert metrics["structured_near_miss_status"] == "PARSER_DERIVED_NOT_GATE_ELIGIBLE"
    metrics["local_gradients_finite_nonzero"] = True
    gates = acceptance_gates("late_interaction", metrics, v1_global_r1=1.0, v1_global_r5=1.0)
    assert not gates["structured_near_miss_human_verified"]


def test_c_gate_requires_mask_evidence_and_human_structured_labels() -> None:
    metrics = experiment_metrics(*_scores(), top_n=2, captions=["a house appeared", "a road appeared"], structured_labels_human_verified=True, mask_logits=torch.tensor([[[8.0]]]), mask_targets=torch.tensor([[[1.0]]]))
    metrics["local_gradients_finite_nonzero"] = True
    metrics["faithful_mask"] = True
    gates = acceptance_gates("mask_grounding", metrics, v1_global_r1=1.0, v1_global_r5=1.0)
    assert "structured_near_miss_human_verified" in gates


def test_margin_family_reports_keep_parser_labels_non_gate_eligible() -> None:
    global_scores, local, reranked, relevance, mapping = _scores()
    reports = margin_family_reports(
        local, reranked, mapping, relevance,
        ["a house appeared on the left", "a road disappeared on the right"],
        global_scores, top_n=2, query_groups={"changed_only": torch.tensor([True, True])},
    )
    assert reports["all"]["structured_label_provenance"] == "parser_derived_not_gate_eligible"
    assert reports["changed_only"]["query_count"] == 2
    assert "wrong_object" in reports["all"]["local_structured_near_miss_by_category"]


def test_parser_near_miss_aggregates_pair_captions_and_excludes_ambiguous_direction():
    from land_change_detection.models.qcpr_v3_experiment import parser_derived_near_miss_masks
    captions = ["a house appeared on the left", "a house disappeared on the left", "a house appeared on the right"]
    mapping = torch.tensor([0, 1, 1])
    global_scores = torch.tensor([[.9, .8], [.9, .8], [.9, .8]])
    broad = torch.tensor([[True, False], [True, False], [True, False]])
    masks = parser_derived_near_miss_masks(captions, mapping, global_scores, broad, top_n=2)
    # Candidate pair 1 carries contradictory appeared/disappeared captions.
    assert not masks["wrong_direction"][:, 1].any()
