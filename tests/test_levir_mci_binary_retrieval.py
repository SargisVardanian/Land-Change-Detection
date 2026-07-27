from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_mask(path: Path, on: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=255 if on else 0).save(path)


def _seed_official_levir_mci(root: Path) -> None:
    payload = {}
    for split, colors in {
        "train": [("0001", (255, 0, 0), (0, 255, 0), True), ("0002", (0, 0, 255), (255, 255, 0), False)],
        "val": [("0003", (125, 10, 10), (10, 125, 10), True)],
        "test": [("0004", (5, 40, 90), (90, 40, 5), True)],
    }.items():
        for sample_id, before_color, after_color, changed in colors:
            _write_rgb(root / "images" / split / "A" / f"{sample_id}.png", before_color)
            _write_rgb(root / "images" / split / "B" / f"{sample_id}.png", after_color)
            _write_mask(root / "images" / split / "label" / f"{sample_id}.png", changed)
            payload[sample_id] = [f"{split} sample {sample_id} changed"]
    (root / "LevirCCcaptions.json").write_text(json.dumps(payload), encoding="utf-8")


def test_levir_mci_validate_render_train_eval(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "rs_change_project"
    data_root = project_root / "datasets" / "raw" / "LEVIR-MCI"
    _seed_official_levir_mci(data_root)

    env = {"PYTHONPATH": "src"}
    setup = subprocess.run(
        [sys.executable, "scripts/setup_rs_change_project.py", "--root", str(project_root), "--write-manifest"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert setup.returncode == 0, setup.stderr

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/validate_levir_mci_dataset.py",
            "--project-root",
            str(project_root),
            "--data-root",
            str(data_root),
            "--check-all",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert validate.returncode == 0, validate.stderr
    assert (project_root / "indexes" / "levir_mci_samples.jsonl").exists()

    render = subprocess.run(
        [
            sys.executable,
            "scripts/render_levir_mci_samples.py",
            "--project-root",
            str(project_root),
            "--data-root",
            str(data_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert render.returncode == 0, render.stderr
    assert (project_root / "runs" / "levir_mci_grid.png").exists()

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_levir_mci_binary_retrieval.py",
            "--project-root",
            str(project_root),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(project_root / "runs" / "levir_mci_overfit"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "32",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert train.returncode == 0, train.stderr
    assert (project_root / "runs" / "levir_mci_overfit" / "best.pt").exists()

    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/eval_levir_mci_binary_retrieval.py",
            "--project-root",
            str(project_root),
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(project_root / "runs" / "levir_mci_overfit" / "best.pt"),
            "--output",
            str(project_root / "runs" / "levir_mci_overfit" / "test_metrics.json"),
            "--image-size",
            "32",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    metrics = json.loads((project_root / "runs" / "levir_mci_overfit" / "test_metrics.json").read_text(encoding="utf-8"))
    assert "dice" in metrics
    assert "recall@5" in metrics
