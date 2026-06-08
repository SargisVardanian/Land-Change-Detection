from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_prithvi_semantic_manifest(tmp_path: Path):
    input_manifest = tmp_path / "semantic.jsonl"
    input_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "a",
                        "dataset_name": "SECOND-CC",
                        "task": "semantic_transition_segmentation",
                        "before_path": "/tmp/a_before.png",
                        "after_path": "/tmp/a_after.png",
                        "semantic_before_path": "/tmp/a_sb.png",
                        "semantic_after_path": "/tmp/a_sa.png",
                        "metadata": {
                            "before_ms_path": "/tmp/a_before.npy",
                            "after_ms_path": "/tmp/a_after.npy",
                        },
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "b",
                        "dataset_name": "SECOND-CC",
                        "task": "semantic_transition_segmentation",
                        "before_path": "/tmp/b_before.png",
                        "after_path": "/tmp/b_after.png",
                        "metadata": {},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prithvi.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_prithvi_semantic_manifest.py",
            "--input-manifest",
            str(input_manifest),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["input_contract"] == "prithvi_6band_from_13band"


def test_semantic_manifest_dataset_loads_prithvi_6band_npy(tmp_path: Path):
    from land_change_detection.training.semantic_manifest_dataset import SemanticManifestDataset, load_semantic_manifest

    before_rgb = tmp_path / "before.png"
    after_rgb = tmp_path / "after.png"
    import PIL.Image

    PIL.Image.new("RGB", (8, 8), color=(255, 0, 0)).save(before_rgb)
    PIL.Image.new("RGB", (8, 8), color=(0, 255, 0)).save(after_rgb)
    before_ms = tmp_path / "before.npy"
    after_ms = tmp_path / "after.npy"
    np.save(before_ms, np.random.rand(13, 8, 8).astype(np.float32))
    np.save(after_ms, np.random.rand(13, 8, 8).astype(np.float32))
    manifest = tmp_path / "semantic.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "x",
                "dataset_name": "SECOND-CC",
                "task": "semantic_transition_segmentation",
                "before_path": str(before_rgb),
                "after_path": str(after_rgb),
                "before_ms_path": str(before_ms),
                "after_ms_path": str(after_ms),
                "change_mask_path": None,
                "semantic_before_path": None,
                "semantic_after_path": None,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SemanticManifestDataset(load_semantic_manifest(manifest), input_channels=6)
    item = dataset[0]
    assert tuple(item["before"].shape) == (6, 8, 8)
    assert tuple(item["after"].shape) == (6, 8, 8)
