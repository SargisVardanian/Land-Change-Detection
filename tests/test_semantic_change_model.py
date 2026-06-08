from __future__ import annotations

import torch
import pytest

from land_change_detection.models.semantic_change import (
    SemanticChangeModelConfig,
    build_semantic_change_model,
    encode_transition_map,
)
from land_change_detection.training.losses import semantic_change_loss


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
