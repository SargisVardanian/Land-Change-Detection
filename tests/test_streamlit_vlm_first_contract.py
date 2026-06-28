from __future__ import annotations

from pathlib import Path


def test_app_no_longer_uses_semantic_rows_for_user_answer():
    app_source = Path("app.py").read_text()

    assert "build_cell_report_rows_from_segmentation" not in app_source
    assert "build_pairwise_visual_context" not in app_source
    assert "build_scene_overview_from_rows" not in app_source
    assert "Full A1..D4 technical table" not in app_source
    assert "Pass semantic hints into model" not in app_source
    assert "fallback_summary=None" in app_source


def test_app_sends_contact_sheet_to_vlm():
    app_source = Path("app.py").read_text()

    assert "build_cell_contact_sheet" in app_source
    assert "model_auxiliary_images = [cell_contact_sheet]" in app_source
    assert "[cell_contact_sheet, change_guide_image]" not in app_source
    assert "A1..D4 before/after contact sheet" in app_source
    assert "semantic_context = build_vlm_semantic_context" not in app_source


def test_normal_vlm_selector_is_gemma_only_until_debug_enabled():
    app_source = Path("app.py").read_text()

    assert '"gemma4:e4b (Ollama)"' in app_source
    assert "available_model_presets(show_debug: bool = False)" in app_source
    assert "if show_debug:" in app_source
    assert "reasoning_options = [\"full_local\", \"efficient\"] if explanation_backend == \"Ollama vision\" else [\"efficient\"]" in app_source


def test_normal_app_hides_retired_visual_baseline_segmentation_and_feature_encoder_controls():
    app_source = Path("app.py").read_text()

    assert "land_change_detection.retired_visual_feature_extractor_features" not in app_source
    assert "retired visual feature extractor feature-change diagnostics" not in app_source
    assert "Running retired visual feature extractor SAT feature diagnostics" not in app_source
    assert "retired visual feature extractor feature encoder" not in app_source
    assert "retired visual feature extractor device" not in app_source
    assert "retired_visual_feature_extractor_feature_regions" not in app_source
    assert "timm/vit_base_patch16_retired_visual_feature_extractor.lvd1689m" not in app_source
    assert "timm/vit_small_patch16_retired_visual_feature_extractor.lvd1689m" not in app_source
