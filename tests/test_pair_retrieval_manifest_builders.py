from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)


def _write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_fake_second_like_manifest_builder(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "SECOND-CC"
    _write_rgb(root / "tile001_A.png")
    _write_rgb(root / "tile001_B.png")
    _write_mask(root / "tile001_sem1.png", 1)
    _write_mask(root / "tile001_sem2.png", 2)
    output = tmp_path / "secondcc.jsonl"
    diag = tmp_path / "diag.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_secondcc_pair_retrieval_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--diagnostics-output",
            str(diag),
            "--num-classes",
            "4",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "SECOND-CC"
    assert "dominant_transition" in rows[0]
    assert rows[0]["metadata"]["curriculum_stage"] == "stage_3_transition_aware"
    assert rows[0]["metadata"]["transition_label"] == rows[0]["dominant_transition"]


def test_fake_hiucd_like_manifest_builder(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "Hi-UCD"
    before_dir = root / "before"
    after_dir = root / "after"
    sem_before_dir = root / "sem_before"
    sem_after_dir = root / "sem_after"
    _write_rgb(before_dir / "tile001.png")
    _write_rgb(after_dir / "tile001.png")
    _write_mask(sem_before_dir / "tile001.png", 1)
    _write_mask(sem_after_dir / "tile001.png", 2)
    output = tmp_path / "hiucd.jsonl"
    diag = tmp_path / "diag_hiucd.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_hiucd_pair_retrieval_manifest.py",
            "--root",
            str(root),
            "--before-dir",
            str(before_dir),
            "--after-dir",
            str(after_dir),
            "--semantic-before-dir",
            str(sem_before_dir),
            "--semantic-after-dir",
            str(sem_after_dir),
            "--output",
            str(output),
            "--diagnostics-output",
            str(diag),
            "--num-classes",
            "4",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "Hi-UCD"
    assert "transition_histogram" in rows[0]
    assert rows[0]["metadata"]["curriculum_stage"] == "stage_3_transition_aware"
