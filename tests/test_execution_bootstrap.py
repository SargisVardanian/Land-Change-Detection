from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_change_retrieval_train_eval(tmp_path: Path):
    manifest = tmp_path / "toy.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "item_id": "a",
                        "mode": "pair_analog",
                        "features": [1.0, 0.0],
                        "transition_label": "cropland->built_up",
                        "positives": ["b"],
                        "negatives": ["c"],
                    }
                ),
                json.dumps(
                    {
                        "item_id": "b",
                        "mode": "pair_analog",
                        "features": [0.9, 0.1],
                        "transition_label": "cropland->built_up",
                        "positives": ["a"],
                        "negatives": ["c"],
                    }
                ),
                json.dumps(
                    {
                        "item_id": "c",
                        "mode": "pair_analog",
                        "features": [0.0, 1.0],
                        "transition_label": "water->wetland",
                        "positives": [],
                        "negatives": ["a"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_change_retrieval_train_eval.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "eval_metrics.json").exists()


def test_build_levir_cc_text_index_numpy_fallback(tmp_path: Path):
    manifest = tmp_path / "text_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "a", "features": [1.0, 0.0], "text": "new building", "transition_label": "x"}),
                json.dumps({"item_id": "b", "features": [0.0, 1.0], "text": "water expands", "transition_label": "y"}),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "index_out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_levir_cc_text_index.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["count"] == 2
    assert metadata["dim"] == 2
    assert metadata["backend"] in {"faiss", "numpy"}
