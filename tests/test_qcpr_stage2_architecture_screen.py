from __future__ import annotations

import torch

from screen_qcpr_stage2_compatible_architectures import (
    FramewiseGatedDifferenceFusion,
    ScreenModel,
    SinglePassConfig,
    TextConditionedTemporalFusion,
)


def _config() -> SinglePassConfig:
    return SinglePassConfig(
        visual_dim=12,
        text_dim=8,
        retrieval_dim=8,
        grid_size=4,
        pair_heads=4,
        text_heads=2,
        dropout=0.0,
    )


def test_framewise_and_text_conditioned_fusion_shapes() -> None:
    frames = torch.randn(3, 2, 16, 12)
    text = torch.randn(3, 8)
    assert FramewiseGatedDifferenceFusion(12)(frames).shape == (3, 16, 12)
    assert TextConditionedTemporalFusion(12, 8)(frames, text).shape == (3, 16, 12)


def test_b2_fusion_changes_when_query_changes() -> None:
    torch.manual_seed(7)
    frames = torch.randn(2, 2, 16, 12)
    fusion = TextConditionedTemporalFusion(12, 8).eval()
    first = fusion(frames, torch.zeros(2, 8))
    second = fusion(frames, torch.ones(2, 8))
    assert not torch.allclose(first, second)


def test_b0_b1_b2_screen_heads_have_finite_gradients() -> None:
    torch.manual_seed(8)
    frames = torch.randn(3, 2, 16, 12)
    base = torch.randn(3, 8)
    tokens = torch.randn(3, 5, 8)
    attention = torch.ones(3, 5, dtype=torch.bool)
    content = attention.clone()
    for kind in ("B0", "B1", "B2"):
        model = ScreenModel(kind, _config())
        visual = frames[:, 0] if kind == "B0" else frames
        scores = model(visual, base, tokens, attention, content)
        assert scores.shape == (3, 3)
        loss = scores.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients
        assert all(torch.isfinite(grad).all() for grad in gradients)


def test_b1_temporal_fusion_receives_gradient_through_pair_adapter() -> None:
    torch.manual_seed(11)
    model = ScreenModel("B1", _config())
    frames = torch.randn(3, 2, 16, 12)
    base = torch.randn(3, 8)
    tokens = torch.randn(3, 5, 8)
    attention = torch.ones(3, 5, dtype=torch.bool)
    scores = model(frames, base, tokens, attention, attention)
    scores.square().mean().backward()
    gradient = model.temporal.mix[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_score_matrix_supports_multiple_captions_per_pair() -> None:
    torch.manual_seed(9)
    model = ScreenModel("B2", _config())
    visual = torch.randn(4, 2, 16, 12)
    base = torch.randn(8, 8)
    tokens = torch.randn(8, 5, 8)
    attention = torch.ones(8, 5, dtype=torch.bool)
    content = attention.clone()
    scores = model(visual, base, tokens, attention, content)
    assert scores.shape == (8, 4)
