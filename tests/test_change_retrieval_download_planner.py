from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from land_change_detection.data.dataset_registry import build_download_plan


def test_planner_refuses_dynamicearthnet_in_baseline_mode(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    plan = build_download_plan(project_root, phase="baseline", reserve_gb=120, max_download_gb=50)
    names = {row["canonical_name"]: row for row in plan["entries"]}
    assert "DynamicEarthNet" not in names


def test_planner_detects_existing_levir_and_avoids_duplicate_download(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    existing = project_root / "datasets" / "raw" / "LEVIR-MCI-unpacked" / "LEVIR-MCI-dataset" / "images"
    existing.mkdir(parents=True, exist_ok=True)
    plan = build_download_plan(project_root, phase="baseline", reserve_gb=120, max_download_gb=50)
    levir = next(row for row in plan["entries"] if row["canonical_name"] == "LEVIR-MCI")
    assert levir["exists"] is True
    assert levir["planned_download_bytes"] == 0


def test_planner_blocks_unknown_size_automatic_downloads_by_default(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    plan = build_download_plan(project_root, phase="baseline", reserve_gb=120, max_download_gb=50)
    second_cc = next(row for row in plan["entries"] if row["canonical_name"] == "SECOND-CC")
    assert second_cc["allowed"] is False
    assert "unknown_size_requires_allow_unknown_size" in second_cc["blocked_reasons"]


def test_planner_cli_writes_json(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    output = tmp_path / "download_plan.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_change_retrieval_downloads.py",
            "--project-root",
            str(project_root),
                "--phase",
                "baseline",
                "--reserve-gb",
                "1",
                "--max-download-gb",
                "50",
            "--output-json",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["phase"] == "baseline"
    assert any(row["canonical_name"] == "LEVIR-MCI" for row in payload["entries"])
