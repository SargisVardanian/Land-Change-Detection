from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_semantic_change_task_manifests(tmp_path: Path):
    index_path = tmp_path / "second_cc_samples.jsonl"
    rows = [
        {
            "sample_id": "tile42",
            "dataset_name": "SECOND-CC",
            "before_path": "/tmp/tile42_A.png",
            "after_path": "/tmp/tile42_B.png",
            "mask_path": "/tmp/tile42_mask.png",
            "semantic_before_path": "/tmp/tile42_label1.png",
            "semantic_after_path": "/tmp/tile42_label2.png",
            "caption": "cropland became built-up",
            "split": "val",
            "metadata": {},
        },
        {
            "sample_id": "tile77",
            "dataset_name": "SECOND-CC",
            "before_path": "/tmp/tile77_A.png",
            "after_path": "/tmp/tile77_B.png",
            "mask_path": "/tmp/tile77_mask.png",
            "semantic_before_path": None,
            "semantic_after_path": None,
            "caption": "water expanded",
            "split": "test",
            "metadata": {},
        },
    ]
    index_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output_dir = tmp_path / "semantic"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_semantic_change_task_manifests.py",
            "--index",
            str(index_path),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "SECOND-CC",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr

    mask_manifest = output_dir / "second_cc_change_mask_manifest.jsonl"
    transition_manifest = output_dir / "second_cc_transition_manifest.jsonl"
    assert mask_manifest.exists()
    assert transition_manifest.exists()

    mask_rows = [json.loads(line) for line in mask_manifest.read_text(encoding="utf-8").splitlines()]
    transition_rows = [json.loads(line) for line in transition_manifest.read_text(encoding="utf-8").splitlines()]
    assert len(mask_rows) == 2
    assert len(transition_rows) == 1
    assert transition_rows[0]["task"] == "semantic_transition_segmentation"
