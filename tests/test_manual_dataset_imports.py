from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_import_manual_second_cc(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "manual" / "SECOND-CC"
    _write_rgb(root / "tile001_A.png")
    _write_rgb(root / "tile001_B.png")
    _write_mask(root / "tile001_sem1.png", 1)
    _write_mask(root / "tile001_sem2.png", 2)

    result = subprocess.run(
        [sys.executable, "scripts/import_manual_second_cc.py", "--project-root", str(project_root), "--num-classes", "4"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (project_root / "indexes" / "manual_second_cc_samples.jsonl").exists()
    assert (project_root / "indexes" / "manual_second_cc_pair_manifest.jsonl").exists()
    assert (root / "IMPORT_PROVENANCE.json").exists()
    assert (project_root / "datasets" / "raw" / "SECOND-CC").exists()


def test_import_manual_hi_ucd(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "manual" / "Hi-UCD"
    _write_rgb(root / "before" / "tile001.png")
    _write_rgb(root / "after" / "tile001.png")
    _write_mask(root / "sem_before" / "tile001.png", 1)
    _write_mask(root / "sem_after" / "tile001.png", 2)

    result = subprocess.run(
        [sys.executable, "scripts/import_manual_hi_ucd.py", "--project-root", str(project_root), "--num-classes", "4"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (project_root / "indexes" / "manual_hi_ucd_samples.jsonl").exists()
    assert (project_root / "indexes" / "manual_hi_ucd_pair_manifest.jsonl").exists()
    assert (root / "IMPORT_PROVENANCE.json").exists()


def test_import_manual_qag360k(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "manual" / "QAG-360K"
    _write_rgb(root / "images" / "img001.png")
    (root / "annotations.json").write_text(
        json.dumps([{"id": "q1", "question": "What changed?", "answer": "road", "image": "images/img001.png"}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/import_manual_qag360k.py", "--project-root", str(project_root)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line)
        for line in (project_root / "indexes" / "manual_qag360k_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["dataset_name"] == "QAG-360K"
    assert rows[0]["question"] == "What changed?"


def test_import_manual_terra_cd(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "manual" / "TERRA-CD"
    _write_rgb(root / "train" / "scene01_before.png")
    _write_rgb(root / "train" / "scene01_after.png")
    _write_mask(root / "train" / "scene01_sem1.png", 1)
    _write_mask(root / "train" / "scene01_sem2.png", 2)

    result = subprocess.run(
        [sys.executable, "scripts/import_manual_terra_cd.py", "--project-root", str(project_root), "--num-classes", "4"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (project_root / "indexes" / "manual_terra_cd_samples.jsonl").exists()
    assert (project_root / "indexes" / "manual_terra_cd_pair_manifest.jsonl").exists()
