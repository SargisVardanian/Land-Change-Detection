from __future__ import annotations

from pathlib import Path


def test_app_no_longer_uses_semantic_rows_for_user_answer():
    app_source = Path("app.py").read_text()

    assert "build_cell_report_rows_from_segmentation" not in app_source
    assert "build_pairwise_visual_context" not in app_source
    assert "build_scene_overview_from_rows" not in app_source
    assert "Full A1..D4 technical table" not in app_source
    assert "Pass semantic hints into model" not in app_source
    assert "build_visual_fallback_summary" in app_source


def test_app_sends_contact_sheet_to_vlm():
    app_source = Path("app.py").read_text()

    assert "build_cell_contact_sheet" in app_source
    assert "model_auxiliary_images = [cell_contact_sheet, change_guide_image]" in app_source
    assert "A1..D4 before/after contact sheet" in app_source
