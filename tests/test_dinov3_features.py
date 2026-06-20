from pathlib import Path

import pytest

timm = pytest.importorskip("timm")

from land_change_detection.dinov3_features import (
    DINO_V3_FACEBOOK_VITL16_SAT,
    DINO_V3_VITL16_SAT,
    _interpret_feature_shift,
    model_dir_complete,
)


def test_dinov3_keeps_only_sat_large_debug_model() -> None:
    assert DINO_V3_FACEBOOK_VITL16_SAT == "facebook/dinov3-vitl16-pretrain-sat493m"
    assert DINO_V3_VITL16_SAT == "timm/vit_large_patch16_dinov3.sat493m"


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
