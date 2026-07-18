from __future__ import annotations

import torch

from audit_a0_fusion import _metrics


def _artifact(scores: torch.Tensor) -> dict:
    return {
        "global_scores": scores,
        "relevance": torch.tensor([[1, 0, 0], [0, 1, 0]]),
        "query_groups": {"changed_only": torch.tensor([True, True])},
    }


def test_changed_recall_is_the_fusion_feasibility_metric() -> None:
    scores = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    metrics = _metrics(scores, _artifact(scores))
    assert metrics["candidate_recall_at_100"] == 1.0
    assert metrics["semantic_ndcg_at_10"] > 0.0


def test_fusion_inputs_preserve_one_shared_score_matrix_shape() -> None:
    baseline = torch.randn(5, 7)
    trained = torch.randn(5, 7)
    fused = 0.5 * baseline + 0.5 * trained
    assert fused.shape == baseline.shape == trained.shape

