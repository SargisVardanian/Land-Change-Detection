from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_levir_cc_pair_manifest_and_overfit(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "LEVIR-CC"
    _write_rgb(root / "train" / "pair_a_before.png", (255, 0, 0))
    _write_rgb(root / "train" / "pair_a_after.png", (0, 255, 0))
    _write_rgb(root / "train" / "pair_b_before.png", (250, 10, 10))
    _write_rgb(root / "train" / "pair_b_after.png", (10, 250, 10))
    _write_rgb(root / "train" / "pair_c_before.png", (0, 0, 255))
    _write_rgb(root / "train" / "pair_c_after.png", (255, 255, 0))
    (root / "captions.json").write_text(
        json.dumps(
            [
                {"id": "pair_a", "caption": "new red building appears", "transition_label": "urban_growth"},
                {"id": "pair_a", "caption": "urban expansion near the road", "transition_label": "urban_growth"},
                {"id": "pair_b", "caption": "another red building appears", "transition_label": "urban_growth"},
                {"id": "pair_c", "caption": "water expands strongly", "transition_label": "water_expansion"},
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "levir_cc_pair_manifest.jsonl"
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_levir_cc_pair_manifest.py",
            "--root",
            str(root),
            "--output",
            str(manifest),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert build.returncode == 0, build.stderr
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert rows[0]["dataset_name"] == "LEVIR-CC"
    assert rows[0]["before_path"].endswith("_before.png")
    assert rows[0]["pair_id"] == "pair_a"
    assert rows[0]["split"] == "train"
    assert sum(1 for row in rows if row["pair_id"] == "pair_a") == 2

    output_dir = tmp_path / "overfit_run"
    overfit = subprocess.run(
        [
            sys.executable,
            "scripts/overfit_levir_cc_retrieval_100.py",
            "--levir-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--preset",
            "simple_patch_smoke",
            "--epochs",
            "6",
            "--batch-size",
            "3",
            "--image-size",
            "32",
            "--max-train-samples",
            "4",
            "--device",
            "cpu",
            "--min-r5-improvement",
            "0.0",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert overfit.returncode == 0, overfit.stderr
    report = json.loads((output_dir / "overfit_report.json").read_text(encoding="utf-8"))
    assert report["improved_recall@5"] is True
    assert (output_dir / "best.pt").exists()


def test_validate_levir_cc_pair_manifest_rejects_pair_leakage(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "leak.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "pair_a#cap000",
                "pair_id": "pair_a",
                "before_path": str(tmp_path / "train" / "pair_a_before.png"),
                "after_path": str(tmp_path / "train" / "pair_a_after.png"),
                "caption": "train caption",
                "split": "train",
                "metadata": {"transition_label": "urban_growth"},
            },
            {
                "sample_id": "pair_a#cap001",
                "pair_id": "pair_a",
                "before_path": str(tmp_path / "val" / "pair_a_before.png"),
                "after_path": str(tmp_path / "val" / "pair_a_after.png"),
                "caption": "val caption",
                "split": "val",
                "metadata": {"transition_label": "urban_growth"},
            },
        ],
    )
    validate = subprocess.run(
        [
            sys.executable,
            "scripts/validate_levir_cc_pair_manifest.py",
            "--manifest",
            str(manifest),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert validate.returncode != 0
