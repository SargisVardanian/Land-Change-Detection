from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from land_change_detection.change_retrieval_datasets import LEVIRMciDataset, SecondCCDataset, load_change_samples


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=255).save(path)


def test_setup_rs_change_project_writes_layout_and_manifest(tmp_path: Path):
    root = tmp_path / "rs_change_project"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_rs_change_project.py",
            "--root",
            str(root),
            "--write-manifest",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (root / "datasets" / "raw").exists()
    assert (root / "datasets" / "dataset_manifest.md").exists()


def test_index_and_render_change_dataset_roundtrip(tmp_path: Path):
    root = tmp_path / "LEVIR-MCI"
    _write_rgb(root / "train" / "sample1_before.png", (255, 0, 0))
    _write_rgb(root / "train" / "sample1_after.png", (0, 255, 0))
    _write_mask(root / "train" / "sample1_mask.png")
    (root / "captions.json").write_text(json.dumps({"sample1": "new buildings appeared"}), encoding="utf-8")

    index_path = tmp_path / "index.jsonl"
    render_path = tmp_path / "preview.png"

    index_result = subprocess.run(
        [
            sys.executable,
            "scripts/index_change_retrieval_dataset.py",
            "--dataset-name",
            "LEVIR-MCI",
            "--root",
            str(root),
            "--output",
            str(index_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert index_result.returncode == 0, index_result.stderr

    samples = load_change_samples(index_path)
    assert len(samples) == 1
    assert samples[0].caption == "new buildings appeared"

    render_result = subprocess.run(
        [
            sys.executable,
            "scripts/render_change_retrieval_sample.py",
            "--index",
            str(index_path),
            "--sample-id",
            "sample1",
            "--output",
            str(render_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert render_result.returncode == 0, render_result.stderr
    assert render_path.exists()


def test_change_dataset_classes_from_root(tmp_path: Path):
    root = tmp_path / "SECOND-CC"
    _write_rgb(root / "val" / "tile42_A.png", (1, 2, 3))
    _write_rgb(root / "val" / "tile42_B.png", (3, 2, 1))
    _write_mask(root / "val" / "tile42_mask.png")

    levir_root = tmp_path / "LEVIR-MCI"
    _write_rgb(levir_root / "train" / "scene9_before.png", (10, 20, 30))
    _write_rgb(levir_root / "train" / "scene9_after.png", (30, 20, 10))

    second = SecondCCDataset.from_root(root)
    levir = LEVIRMciDataset.from_root(levir_root)

    assert len(second) == 1
    assert len(levir) == 1
    assert second[0]["sample_id"] == "tile42"
    assert levir[0]["sample_id"] == "scene9"


def test_build_levir_cc_manifest(tmp_path: Path):
    captions_path = tmp_path / "captions.json"
    captions_path.write_text(
        json.dumps(
            [
                {"id": "a", "caption": "new building appears", "transition_label": "cropland->built_up"},
                {"id": "b", "caption": "another building appears", "transition_label": "cropland->built_up"},
                {"id": "c", "caption": "water expands", "transition_label": "dryland->water"},
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_levir_cc_manifest.py",
            "--captions-json",
            str(captions_path),
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
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["mode"] == "text_bitemporal"
    assert len(rows[0]["features"]) == 8
    assert "b" in rows[0]["positives"]
