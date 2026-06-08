from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_download_script_skip_modes(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_change_retrieval_datasets.py",
            "--project-root",
            str(project_root),
            "--skip-hf",
            "--skip-second-cc",
            "--skip-reference-repos",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (project_root / "datasets" / "raw").exists()
    assert (project_root / "code").exists()


def test_build_change_retrieval_training_manifest(tmp_path: Path):
    index_path = tmp_path / "index.jsonl"
    rows = [
        {
            "sample_id": "a",
            "dataset_name": "LEVIR-MCI",
            "before_path": "/tmp/a_before.png",
            "after_path": "/tmp/a_after.png",
            "caption": "new buildings appeared",
            "metadata": {"transition_label": "cropland->built_up"},
        },
        {
            "sample_id": "b",
            "dataset_name": "LEVIR-MCI",
            "before_path": "/tmp/b_before.png",
            "after_path": "/tmp/b_after.png",
            "caption": "more new buildings appeared",
            "metadata": {"transition_label": "cropland->built_up"},
        },
        {
            "sample_id": "c",
            "dataset_name": "LEVIR-MCI",
            "before_path": "/tmp/c_before.png",
            "after_path": "/tmp/c_after.png",
            "caption": "water expanded",
            "metadata": {"transition_label": "dryland->water"},
        },
    ]
    index_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output_path = tmp_path / "train_manifest.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_change_retrieval_training_manifest.py",
            "--index",
            str(index_path),
            "--output",
            str(output_path),
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

    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(output_rows) == 3
    assert output_rows[0]["mode"] == "pair_analog"
    assert "b" in output_rows[0]["positives"]
    assert len(output_rows[0]["features"]) == 8
