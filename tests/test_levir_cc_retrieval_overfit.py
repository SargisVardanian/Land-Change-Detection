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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    rows = _read_jsonl(manifest)
    assert len(rows) == 3
    assert rows[0]["dataset_name"] == "LEVIR-CC"
    assert rows[0]["before_path"].endswith("_before.png")
    assert rows[0]["pair_id"] == "pair_a"
    assert rows[0]["split"] == "train"
    assert rows[0]["split_source"] == "official_layout"
    assert len(rows[0]["captions"]) == 2
    caption_manifest = tmp_path / "levir_cc_caption_queries.jsonl"
    caption_rows = _read_jsonl(caption_manifest)
    assert len(caption_rows) == 4
    assert caption_rows[0]["caption_index"] == 0
    assert caption_rows[0]["split_source"] == "official_layout"
    train_rows = _read_jsonl(tmp_path / "levir_cc_caption_queries_train.jsonl")
    val_rows = _read_jsonl(tmp_path / "levir_cc_caption_queries_val.jsonl")
    test_rows = _read_jsonl(tmp_path / "levir_cc_caption_queries_test.jsonl")
    overfit_rows = _read_jsonl(tmp_path / "levir_cc_caption_queries_overfit_100.jsonl")
    assert len(train_rows) == 4
    assert len(val_rows) == 0
    assert len(test_rows) == 0
    assert len(overfit_rows) == 4
    build_payload = json.loads(build.stdout)
    assert build_payload["split_source_counts"]["official_layout"] == 3
    assert build_payload["fallback_pair_count"] == 0
    assert build_payload["split_caption_row_counts"]["train"] == 4
    assert build_payload["split_caption_row_counts"]["overfit_100"] == 4

    assert {row["pair_id"] for row in overfit_rows} == {"pair_a", "pair_b", "pair_c"}


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
