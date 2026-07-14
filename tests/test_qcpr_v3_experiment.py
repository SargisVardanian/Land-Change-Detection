from __future__ import annotations

import torch

from land_change_detection.models.qcpr_v3_experiment import acceptance_gates, experiment_metrics


def _scores():
    global_scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    local_scores = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    reranked_scores = global_scores + local_scores
    relevance = torch.tensor([[True, False, False], [False, True, False]])
    mapping = torch.tensor([0, 1])
    return global_scores, local_scores, reranked_scores, relevance, mapping


def test_experiment_metrics_and_b_gate_are_deterministic() -> None:
    metrics = experiment_metrics(*_scores(), top_n=2)
    metrics["local_gradients_finite_nonzero"] = True
    gates = acceptance_gates("late_interaction", metrics, v1_global_r1=1.0, v1_global_r5=1.0)
    assert all(gates.values())
    assert metrics["local_positive_hard_negative_margin"] > 0


def test_c_gate_requires_mask_evidence_without_relaxing_b_gates() -> None:
    metrics = experiment_metrics(*_scores(), top_n=2, mask_logits=torch.tensor([[[8.0]]]), mask_targets=torch.tensor([[[1.0]]]))
    metrics["local_gradients_finite_nonzero"] = True
    metrics["faithful_mask"] = True
    gates = acceptance_gates("mask_grounding", metrics, v1_global_r1=1.0, v1_global_r5=1.0)
    assert all(gates.values())
