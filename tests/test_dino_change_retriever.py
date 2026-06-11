from __future__ import annotations

import pytest
import torch

from land_change_detection.models.dino_change_retriever import (
    DINOV2_PATH_ERROR,
    DINOChangeRetriever,
    DINOChangeRetrieverConfig,
)


def test_simple_patch_forward_pass_runs_without_external_weights():
    model = DINOChangeRetriever(DINOChangeRetrieverConfig(visual_backbone="simple_patch", hidden_dim=96, transformer_heads=6))
    before = torch.randn(2, 3, 224, 224)
    after = torch.randn(2, 3, 224, 224)
    output = model(before, after, ["new building detected", "road widened in the south"])
    assert output["change_embedding"].shape == (2, 96)
    assert output["text_embedding"].shape == (2, 96)
    assert output["patch_tokens"].ndim == 3


def test_dinov2_missing_path_fails_with_clear_message(tmp_path):
    missing = tmp_path / "dinov2-small"
    with pytest.raises(FileNotFoundError, match="DINOv2 model path not found"):
        DINOChangeRetriever(
            DINOChangeRetrieverConfig(
                visual_backbone="dinov2",
                dinov2_model_path=str(missing),
                local_files_only=True,
            )
        )
    assert "simple_patch" in DINOV2_PATH_ERROR
