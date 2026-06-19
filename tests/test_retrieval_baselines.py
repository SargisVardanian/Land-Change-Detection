from __future__ import annotations

from land_change_detection.retrieval_baselines import preset_by_name


def test_supported_simple_patch_smoke_preset_exists():
    preset = preset_by_name("simple_patch_smoke")
    assert preset.supported is True
    assert preset.visual_backbone == "simple_patch"
    assert preset.text_backbone == "simple_text"


def test_remoteclip_pair_fusion_preset_is_registered_but_not_supported_yet():
    preset = preset_by_name("remoteclip_pair_fusion")
    assert preset.supported is False
    assert "not implemented" in preset.notes[-1]
