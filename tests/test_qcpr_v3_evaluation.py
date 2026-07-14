from __future__ import annotations

import pytest
import torch

from land_change_detection.models.qcpr_v3_evaluation import soft_segmentation_metrics, two_stage_retrieval_metrics


def test_two_stage_metrics_report_candidate_loss_and_gain() -> None:
    global_scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    reranked = torch.tensor([[1.0, 4.0, 9.0], [1.0, 2.0, 9.0]])
    relevance = torch.tensor([[False, True, False], [False, False, True]])
    metrics = two_stage_retrieval_metrics(global_scores, reranked, relevance, top_n=2)
    assert metrics["candidate_recall_at_n"] == pytest.approx(0.5)
    assert metrics["positives_lost_before_reranking"] == 1
    assert metrics["conditional_reranking_gain_r1"] > 0


def test_soft_segmentation_reports_empty_nonempty_and_pointing() -> None:
    logits = torch.full((2, 4, 4), -8.0)
    targets = torch.zeros_like(logits, dtype=torch.bool)
    logits[0, 1, 1] = 8.0; targets[0, 1, 1] = True
    metrics = soft_segmentation_metrics(logits, targets)
    assert metrics["nonempty_count"] == 1 and metrics["empty_count"] == 1
    assert metrics["pointing_game"] == 1
    assert metrics["empty_false_positive_area"] == 0
