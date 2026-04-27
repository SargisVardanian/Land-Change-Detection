from pathlib import Path

from land_change_detection.dinov3_features import (
    DINO_V3_FACEBOOK_VITL16_SAT,
    DINO_V3_VITB16,
    DINO_V3_VITL16_SAT,
    DINO_V3_VITS16,
    _artifact_from_region_map,
    _interpret_feature_shift,
    model_dir_complete,
)
import numpy as np


def test_dinov3_open_repo_ids_are_used() -> None:
    assert DINO_V3_FACEBOOK_VITL16_SAT == "facebook/dinov3-vitl16-pretrain-sat493m"
    assert DINO_V3_VITS16.startswith("timm/")
    assert DINO_V3_VITB16.startswith("timm/")
    assert DINO_V3_VITL16_SAT.startswith("timm/")


def test_model_dir_complete_requires_weights_and_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert not model_dir_complete(model_dir)
    (model_dir / "config.json").write_text("{}")
    assert not model_dir_complete(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"fake")
    assert model_dir_complete(model_dir)


def test_interpret_feature_shift_bands() -> None:
    assert "strong" in _interpret_feature_shift(0.25, 1.0)
    assert "moderate" in _interpret_feature_shift(0.15, 1.0)
    assert "weak" in _interpret_feature_shift(0.09, 1.0)
    assert "stable" in _interpret_feature_shift(0.03, 1.0)


def test_dinov3_region_artifact_marks_unsupervised_regions() -> None:
    region_map = np.array([[0, 1], [1, 2]], dtype=np.int32)
    artifact = _artifact_from_region_map(region_map, "timm/example", shared_cluster_space=True)
    assert artifact.metadata is not None
    assert artifact.metadata["backend"] == "dinov3_feature_regions"
    assert artifact.metadata["semantic_classes"] is False
    assert artifact.metadata["shared_cluster_space"] is True
    assert artifact.color_map.shape == (2, 2, 3)
