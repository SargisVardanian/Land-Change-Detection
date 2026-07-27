from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def test_prithvi_terratorch_experimental_runner_fallback(tmp_path: Path):
    before_rgb = tmp_path / "before.png"
    after_rgb = tmp_path / "after.png"
    mask = tmp_path / "mask.png"
    sem_before = tmp_path / "sem_before.png"
    sem_after = tmp_path / "sem_after.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(before_rgb)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(after_rgb)
    Image.new("L", (8, 8), color=255).save(mask)
    Image.new("L", (8, 8), color=1).save(sem_before)
    Image.new("L", (8, 8), color=2).save(sem_after)
    before_ms = tmp_path / "before.npy"
    after_ms = tmp_path / "after.npy"
    np.save(before_ms, np.random.rand(13, 8, 8).astype(np.float32))
    np.save(after_ms, np.random.rand(13, 8, 8).astype(np.float32))

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "p1",
                "dataset_name": "SECOND-CC",
                "task": "semantic_transition_segmentation",
                "before_path": str(before_rgb),
                "after_path": str(after_rgb),
                "before_ms_path": str(before_ms),
                "after_ms_path": str(after_ms),
                "change_mask_path": str(mask),
                "semantic_before_path": str(sem_before),
                "semantic_after_path": str(sem_after),
                "input_contract": "prithvi_6band_from_13band",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_prithvi_terratorch_experimental.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--allow-fallback-baseline",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    status = json.loads((output_dir / "prithvi_experimental_status.json").read_text(encoding="utf-8"))
    assert status["mode"] == "fallback_baseline"
    assert (output_dir / "semantic_change_model.pt").exists()


def test_prithvi_terratorch_experimental_runner_blocks_without_fallback(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join([]), encoding="utf-8")
    output_dir = tmp_path / "blocked"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_prithvi_terratorch_experimental.py",
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
    assert result.returncode != 0
    status = json.loads((output_dir / "prithvi_experimental_status.json").read_text(encoding="utf-8"))
    assert status["mode"] == "blocked_no_fallback"
