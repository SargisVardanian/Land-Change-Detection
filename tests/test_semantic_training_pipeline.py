from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_semantic_training_and_eval_pipeline(tmp_path: Path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    mask = tmp_path / "mask.png"
    sem_before = tmp_path / "sem_before.png"
    sem_after = tmp_path / "sem_after.png"
    _write_rgb(before, (255, 0, 0))
    _write_rgb(after, (0, 255, 0))
    _write_mask(mask, 255)
    _write_mask(sem_before, 1)
    _write_mask(sem_after, 2)

    manifest = tmp_path / "semantic_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "x1",
                "dataset_name": "SECOND-CC",
                "task": "semantic_transition_segmentation",
                "before_path": str(before),
                "after_path": str(after),
                "change_mask_path": str(mask),
                "semantic_before_path": str(sem_before),
                "semantic_after_path": str(sem_after),
                "split": "train",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    train_result = subprocess.run(
        [
            sys.executable,
            "scripts/train_semantic_change.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--input-channels",
            "6",
            "--num-classes",
            "12",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert train_result.returncode == 0, train_result.stderr
    assert (output_dir / "semantic_change_model.pt").exists()
    assert (output_dir / "metrics.json").exists()

    eval_output = output_dir / "eval.json"
    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_semantic_change.py",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(output_dir / "semantic_change_model.pt"),
            "--output",
            str(eval_output),
            "--batch-size",
            "1",
            "--input-channels",
            "6",
            "--num-classes",
            "12",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert eval_result.returncode == 0, eval_result.stderr
    metrics = json.loads(eval_output.read_text(encoding="utf-8"))
    assert metrics["samples"] == 1
    assert "change_accuracy" in metrics


def test_semantic_change_loss_handles_all_ignore(tmp_path: Path):
    import torch

    from land_change_detection.models import SemanticChangeModelConfig, build_semantic_change_model
    from land_change_detection.training.losses import semantic_change_loss

    model = build_semantic_change_model(SemanticChangeModelConfig(input_channels=6, num_classes=4, base_channels=8, encoder_depth=2))
    output = model(torch.randn(1, 6, 8, 8), torch.randn(1, 6, 8, 8))
    before_target = torch.full((1, 8, 8), -1, dtype=torch.long)
    after_target = torch.full((1, 8, 8), -1, dtype=torch.long)
    change_target = torch.zeros(1, 8, 8)

    loss = semantic_change_loss(output, before_target, after_target, change_target=change_target, ignore_index=-1)
    assert torch.isfinite(loss)
