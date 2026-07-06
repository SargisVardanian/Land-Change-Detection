from __future__ import annotations

import torch

from land_change_detection.models.temporal_change_encoder import (
    TemporalChangeEncoder,
    TemporalChangeEncoderConfig,
)


def test_directional_encoder_changes_when_timestamps_are_swapped():
    torch.manual_seed(7)
    config = TemporalChangeEncoderConfig(
        input_dim=8,
        hidden_dim=8,
        depth=1,
        heads=2,
        ffn_dim=16,
        grid_size=2,
        window_size=1,
        global_tokens=2,
        drop_path_max=0.0,
        use_direction_embeddings=True,
        use_explicit_change_fusion=True,
    )
    model = TemporalChangeEncoder(config).eval()
    features = torch.randn(2, 2, 4, 8)
    forward = model(features).pair_embedding
    reverse = model(features.flip(1)).pair_embedding
    assert not torch.allclose(forward, reverse, atol=1e-5, rtol=1e-5)


def test_legacy_encoder_remains_shape_compatible():
    config = TemporalChangeEncoderConfig(
        input_dim=8,
        hidden_dim=8,
        depth=1,
        heads=2,
        ffn_dim=16,
        grid_size=2,
        window_size=1,
        global_tokens=2,
        drop_path_max=0.0,
    )
    model = TemporalChangeEncoder(config).eval()
    output = model(torch.randn(3, 2, 4, 8))
    assert output.per_time_tokens.shape == (3, 2, 4, 8)
    assert output.change_tokens.shape == (3, 4, 8)
    assert output.global_tokens.shape == (3, 2, 8)
    assert output.pair_embedding.shape == (3, 8)
