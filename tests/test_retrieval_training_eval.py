from __future__ import annotations

import json
import subprocess
from pathlib import Path

from land_change_detection.training.metrics import mean_average_precision, recall_at_k, transition_consistency_score
from land_change_detection.training.retrieval_datasets import ManifestRetrievalDataset, load_manifest_records
from land_change_detection.training.samplers import build_triplets


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "item_id": "a",
            "mode": "pair_analog",
            "features": [1.0, 0.0],
            "transition_label": "cropland->built_up",
            "positives": ["b"],
            "negatives": ["c"],
        },
        {
            "item_id": "b",
            "mode": "pair_analog",
            "features": [0.9, 0.1],
            "transition_label": "cropland->built_up",
            "positives": ["a"],
            "negatives": ["c"],
        },
        {
            "item_id": "c",
            "mode": "pair_analog",
            "features": [0.0, 1.0],
            "transition_label": "water->wetland",
            "positives": [],
            "negatives": ["a"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_dataset_and_sampler_roundtrip(tmp_path: Path):
    manifest = tmp_path / "toy.jsonl"
    _write_manifest(manifest)
    dataset = ManifestRetrievalDataset(load_manifest_records(manifest))
    triplets = build_triplets(dataset)

    assert len(dataset) == 3
    assert dataset[0].item_id == "a"
    assert len(triplets) == 2


def test_metrics_smoke():
    assert recall_at_k([False, True], 1) == 0.0
    assert recall_at_k([False, True], 2) == 1.0
    assert mean_average_precision([[True, False], [False, True]]) == 0.75
    assert transition_consistency_score("cropland->built_up", ["cropland->built_up"]) == 1.0


def test_train_and_eval_scripts(tmp_path: Path):
    manifest = tmp_path / "toy.jsonl"
    _write_manifest(manifest)
    output_dir = tmp_path / "train_out"
    eval_out = tmp_path / "eval.json"

    train_cmd = [
        str(Path(".venv/bin/python")),
        "scripts/train_retrieval_head.py",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
    ]
    eval_cmd = [
        str(Path(".venv/bin/python")),
        "scripts/eval_retrieval.py",
        "--manifest",
        str(manifest),
        "--output",
        str(eval_out),
    ]

    subprocess.run(train_cmd, check=True)
    subprocess.run(eval_cmd, check=True)

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "checkpoint.json").exists()
    assert eval_out.exists()
