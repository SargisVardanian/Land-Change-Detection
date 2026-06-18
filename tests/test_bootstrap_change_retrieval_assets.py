from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=255).save(path)


def test_bootstrap_change_retrieval_assets(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    raw = project_root / "datasets" / "raw"

    _write_rgb(raw / "LEVIR-MCI" / "train" / "sample1_before.png", (255, 0, 0))
    _write_rgb(raw / "LEVIR-MCI" / "train" / "sample1_after.png", (0, 255, 0))
    _write_mask(raw / "LEVIR-MCI" / "train" / "sample1_mask.png")
    (raw / "LEVIR-MCI" / "captions.json").write_text(json.dumps({"sample1": "new buildings appeared"}), encoding="utf-8")

    _write_rgb(raw / "SECOND-CC" / "val" / "tile42_A.png", (1, 2, 3))
    _write_rgb(raw / "SECOND-CC" / "val" / "tile42_B.png", (3, 2, 1))
    _write_mask(raw / "SECOND-CC" / "val" / "tile42_mask.png")

    (raw / "LEVIR-CC").mkdir(parents=True, exist_ok=True)
    (raw / "LEVIR-CC" / "captions.json").write_text(
        json.dumps(
            [
                {"id": "a", "caption": "new building appears", "transition_label": "cropland->built_up"},
                {"id": "b", "caption": "another building appears", "transition_label": "cropland->built_up"},
                {"id": "c", "caption": "water expands", "transition_label": "dryland->water"},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_change_retrieval_assets.py",
            "--project-root",
            str(project_root),
            "--feature-dim",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr

    levir_index = project_root / "indexes" / "levir_mci_samples.jsonl"
    second_index = project_root / "indexes" / "second_cc_samples.jsonl"
    text_manifest = project_root / "indexes" / "levir_cc_text_manifest.jsonl"
    preview_manifest = project_root / "indexes" / "preview_samples.json"

    assert levir_index.exists()
    assert second_index.exists()
    assert text_manifest.exists()
    assert preview_manifest.exists()

    preview = json.loads(preview_manifest.read_text(encoding="utf-8"))
    assert preview["LEVIR-MCI"]["sample_id"] == "sample1"
    assert preview["SECOND-CC"]["sample_id"] == "tile42"
    assert any(row["name"] == "LEVIR-MCI" for row in preview["curriculum"])

    render_result = subprocess.run(
        [
            sys.executable,
            "scripts/render_bootstrap_previews.py",
            "--project-root",
            str(project_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert render_result.returncode == 0, render_result.stderr
    assert (project_root / "runs" / "levir_mci_preview.png").exists()
    assert (project_root / "runs" / "second_cc_preview.png").exists()
