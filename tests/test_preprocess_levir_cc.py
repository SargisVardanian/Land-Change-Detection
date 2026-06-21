from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _run_preprocess(root: Path, project_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_levir_cc.py",
            "--root",
            str(root),
            "--project-root",
            str(project_root),
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )


def test_preprocess_levir_cc_preserves_all_captions_and_is_idempotent(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _write_rgb(root / "train" / "A" / "pair_a.png", (255, 0, 0))
    _write_rgb(root / "train" / "B" / "pair_a.png", (0, 255, 0))
    _write_rgb(root / "val" / "A" / "pair_b.png", (10, 20, 30))
    _write_rgb(root / "val" / "B" / "pair_b.png", (30, 20, 10))
    _write_rgb(root / "test" / "A" / "pair_c.png", (0, 0, 255))
    _write_rgb(root / "test" / "B" / "pair_c.png", (255, 255, 0))
    (root / "LevirCCcaptions.json").write_text(
        json.dumps(
            [
                {"id": "pair_a", "caption": "new building appears", "transition_label": "urban_growth"},
                {"id": "pair_a", "caption": "roadside urban expansion", "transition_label": "urban_growth"},
                {"id": "pair_b", "caption": "water shrinks", "transition_label": "water_change"},
                {"id": "pair_c", "caption": "forest loss", "transition_label": "forest_loss"},
            ]
        ),
        encoding="utf-8",
    )

    first = _run_preprocess(root, project_root)
    assert first.returncode == 0, first.stderr
    second = _run_preprocess(root, project_root)
    assert second.returncode == 0, second.stderr

    train_rows = [
        json.loads(line)
        for line in (project_root / "indexes" / "levir_cc_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in (project_root / "indexes" / "levir_cc_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_rows = [
        json.loads(line)
        for line in (project_root / "indexes" / "levir_cc_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_rows = [
        json.loads(line)
        for line in (project_root / "indexes" / "levir_cc_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads((project_root / "reports" / "levir_cc_preprocess_report.json").read_text(encoding="utf-8"))

    assert len(pair_rows) == 3
    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert len(test_rows) == 1
    assert train_rows[0]["before_path"].startswith("datasets/raw/LEVIR-CC/train/")
    assert train_rows[0]["width"] == 32
    assert train_rows[0]["sample_id"] == "pair_a#cap000"
    assert train_rows[1]["sample_id"] == "pair_a#cap001"
    assert report["caption_row_count"] == 4
    assert report["unique_pair_count"] == 3
    assert report["captions_by_split"] == {"train": 2, "val": 1, "test": 1}


def test_preprocess_levir_cc_rejects_corrupt_images(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _write_rgb(root / "train" / "A" / "pair_a.png", (255, 0, 0))
    (root / "train" / "B" / "pair_a.png").parent.mkdir(parents=True, exist_ok=True)
    (root / "train" / "B" / "pair_a.png").write_bytes(b"not-an-image")
    (root / "captions.json").write_text(json.dumps([{"id": "pair_a", "caption": "broken pair"}]), encoding="utf-8")

    result = _run_preprocess(root, project_root)
    assert result.returncode != 0
    assert "corrupt_files" in result.stdout


def test_preprocess_levir_cc_rejects_pair_level_split_leakage(tmp_path: Path):
    project_root = tmp_path / "rs_change_project"
    root = project_root / "datasets" / "raw" / "LEVIR-CC"
    _write_rgb(root / "train" / "A" / "pair_a.png", (255, 0, 0))
    _write_rgb(root / "train" / "B" / "pair_a.png", (0, 255, 0))
    (root / "captions.json").write_text(
        json.dumps(
            [
                {"id": "pair_a", "caption": "caption one", "split": "train"},
                {"id": "pair_a", "caption": "caption two", "split": "val"},
            ]
        ),
        encoding="utf-8",
    )

    result = _run_preprocess(root, project_root)
    assert result.returncode != 0
    assert "split_leakage" in result.stdout
