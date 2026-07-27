from __future__ import annotations

import torch
import pytest

from land_change_detection.models.semantic_change import (
    SemanticChangeModelConfig,
    build_semantic_change_model,
    encode_transition_map,
)
from land_change_detection.training.losses import semantic_change_loss
from land_change_detection.metrics.semantic_change import binary_change_metrics, semantic_change_metrics


def test_semantic_change_model_outputs_semantic_and_auxiliary_change_logits():
    config = SemanticChangeModelConfig(input_channels=6, num_classes=5, base_channels=8, encoder_depth=3, dropout=0.0)
    model = build_semantic_change_model(config)
    before = torch.randn(2, 6, 32, 32)
    after = torch.randn(2, 6, 32, 32)

    output = model(before, after)

    assert output.before_logits.shape == (2, 5, 32, 32)
    assert output.after_logits.shape == (2, 5, 32, 32)
    assert output.change_logits.shape == (2, 1, 32, 32)
    assert output.before_semantic_map.shape == (2, 32, 32)
    assert output.binary_change_map.shape == (2, 32, 32)
    assert output.transition_map().shape == (2, 32, 32)


def test_semantic_change_model_rejects_wrong_channel_count():
    model = build_semantic_change_model(SemanticChangeModelConfig(input_channels=6, base_channels=8))

    with pytest.raises(ValueError, match="expected 6 input channels"):
        model(torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16))


def test_encode_transition_map_uses_semantic_pair_ids():
    before = torch.tensor([[[1, 2], [3, 4]]])
    after = torch.tensor([[[1, 3], [0, 4]]])

    encoded = encode_transition_map(before, after, num_classes=10)

    assert encoded.tolist() == [[[11, 23], [30, 44]]]


def test_semantic_change_loss_is_differentiable():
    config = SemanticChangeModelConfig(input_channels=6, num_classes=4, base_channels=8, encoder_depth=2, dropout=0.0)
    model = build_semantic_change_model(config)
    output = model(torch.randn(1, 6, 16, 16), torch.randn(1, 6, 16, 16))
    before_target = torch.zeros(1, 16, 16, dtype=torch.long)
    after_target = torch.ones(1, 16, 16, dtype=torch.long)

    loss = semantic_change_loss(output, before_target, after_target)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())


def test_semantic_change_loss_respects_ignore_index_for_derived_change_target():
    config = SemanticChangeModelConfig(input_channels=6, num_classes=4, base_channels=8, encoder_depth=2, dropout=0.0)
    model = build_semantic_change_model(config)
    output = model(torch.randn(1, 6, 8, 8), torch.randn(1, 6, 8, 8))
    before_target = torch.zeros(1, 8, 8, dtype=torch.long)
    after_target = torch.ones(1, 8, 8, dtype=torch.long)
    before_target[:, :2, :2] = -1

    loss = semantic_change_loss(output, before_target, after_target, ignore_index=-1)

    assert torch.isfinite(loss)


class _MetricOutput:
    def __init__(self, before_map: torch.Tensor, after_map: torch.Tensor, change_map: torch.Tensor, num_classes: int):
        self.before_logits = torch.nn.functional.one_hot(before_map, num_classes=num_classes).permute(0, 3, 1, 2).float() * 10.0
        self.after_logits = torch.nn.functional.one_hot(after_map, num_classes=num_classes).permute(0, 3, 1, 2).float() * 10.0
        self.change_logits = change_map.unsqueeze(1).float() * 20.0 - 10.0

    @property
    def before_semantic_map(self) -> torch.Tensor:
        return self.before_logits.argmax(dim=1)

    @property
    def after_semantic_map(self) -> torch.Tensor:
        return self.after_logits.argmax(dim=1)

    @property
    def binary_change_map(self) -> torch.Tensor:
        return (torch.sigmoid(self.change_logits) >= 0.5).to(torch.uint8).squeeze(1)

    def transition_map(self) -> torch.Tensor:
        return encode_transition_map(self.before_semantic_map, self.after_semantic_map, self.before_logits.shape[1])


def test_semantic_change_metrics_score_perfect_prediction():
    before = torch.tensor([[[1, 1], [2, 2]]])
    after = torch.tensor([[[1, 2], [2, 3]]])
    change = (before != after).to(torch.float32)
    output = _MetricOutput(before, after, change, num_classes=4)

    metrics = semantic_change_metrics(
        output,
        {
            "before_target": before,
            "after_target": after,
            "change_target": change,
            "dominant_transition": ["1->1"],
        },
        num_classes=4,
    )

    assert metrics["semantic_mean_iou"] == 1.0
    assert metrics["transition_mean_iou"] == 1.0
    assert metrics["binary_change_f1"] == 1.0
    assert metrics["transition_summary"]["target_counts"] == {"1->1": 1, "1->2": 1, "2->2": 1, "2->3": 1}


def test_semantic_change_metrics_detect_partial_transition_mistake_and_ignore_pixels():
    before_target = torch.tensor([[[1, 1], [2, -1]]])
    after_target = torch.tensor([[[1, 2], [2, 3]]])
    before_pred = torch.tensor([[[1, 1], [2, 0]]])
    after_pred = torch.tensor([[[1, 1], [2, 0]]])
    change = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
    output = _MetricOutput(before_pred, after_pred, torch.zeros(1, 2, 2), num_classes=4)

    metrics = semantic_change_metrics(
        output,
        {
            "before_target": before_target,
            "after_target": after_target,
            "change_target": change,
            "dominant_transition": [""],
        },
        num_classes=4,
    )

    assert 0.0 < metrics["semantic_mean_iou"] < 1.0
    assert 0.0 < metrics["transition_mean_iou"] < 1.0
    assert metrics["transition_summary"]["target_counts"] == {"1->1": 1, "1->2": 1, "2->2": 1}
    assert metrics["binary_change_recall"] == 0.0


def test_binary_change_metrics_precision_recall_f1_iou_dice():
    pred = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)
    target = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)

    metrics = binary_change_metrics(pred, target)

    assert metrics["binary_change_precision"] == 0.5
    assert metrics["binary_change_recall"] == 0.5
    assert metrics["binary_change_f1"] == 0.5
    assert metrics["binary_change_iou"] == pytest.approx(1 / 3)
    assert metrics["binary_change_dice"] == 0.5
