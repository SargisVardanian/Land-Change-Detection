from __future__ import annotations

from pathlib import Path

from land_change_detection.dataset_curriculum import curriculum_summary, datasets_by_stage


def test_datasets_by_stage_includes_grounded_and_transition_stages():
    grouped = datasets_by_stage()
    assert "stage_2_grounded" in grouped
    assert any(spec.name == "LEVIR-MCI" for spec in grouped["stage_2_grounded"])
    assert any(spec.name == "SECOND-CC" for spec in grouped["stage_3_transition_aware"])
    assert any(spec.name == "SpaceNet7" for spec in grouped["stage_5_temporal_later"])


def test_curriculum_summary_marks_default_bootstrap_and_paths(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    levir_root = project_root / "datasets" / "raw" / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset"
    levir_root.mkdir(parents=True, exist_ok=True)

    rows = curriculum_summary(project_root)
    levir = next(row for row in rows if row["name"] == "LEVIR-MCI")
    second = next(row for row in rows if row["name"] == "SECOND-CC")

    assert levir["default_bootstrap"] is True
    assert levir["exists"] is True
    assert str(levir_root) == levir["path"]
    assert second["default_bootstrap"] is False
