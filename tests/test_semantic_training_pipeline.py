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
            "--train-manifest",
            str(manifest),
            "--val-manifest",
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
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics_history.json").exists()
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "semantic_eval_val.json").exists()
    train_metrics = json.loads((output_dir / "semantic_eval_val.json").read_text(encoding="utf-8"))
    assert "semantic_mean_iou" in train_metrics
    assert "transition_mean_iou" in train_metrics
    assert "transition_summary" in train_metrics

    eval_output = output_dir / "eval.json"
    eval_result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_semantic_change.py",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(output_dir / "best.pt"),
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
    assert "before_mean_iou" in metrics
    assert "after_mean_iou" in metrics
    assert "transition_mean_iou" in metrics
    assert "binary_change_f1" in metrics
    assert metrics["transition_summary"]["target_counts"]["1->2"] == 256


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


def test_split_aware_semantic_training_writes_optional_test_metrics(tmp_path: Path):
    train_before = tmp_path / "train_before.png"
    train_after = tmp_path / "train_after.png"
    val_before = tmp_path / "val_before.png"
    val_after = tmp_path / "val_after.png"
    test_before = tmp_path / "test_before.png"
    test_after = tmp_path / "test_after.png"
    mask = tmp_path / "mask.png"
    sem_before = tmp_path / "sem_before.png"
    sem_after = tmp_path / "sem_after.png"
    for path in [train_before, val_before, test_before]:
        _write_rgb(path, (255, 0, 0))
    for path in [train_after, val_after, test_after]:
        _write_rgb(path, (0, 255, 0))
    _write_mask(mask, 255)
    _write_mask(sem_before, 1)
    _write_mask(sem_after, 2)

    def write_manifest(path: Path, sample_id: str, before: Path, after: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "dataset_name": "SECOND-CC",
                    "task": "semantic_transition_segmentation",
                    "before_path": str(before),
                    "after_path": str(after),
                    "change_mask_path": str(mask),
                    "semantic_before_path": str(sem_before),
                    "semantic_after_path": str(sem_after),
                    "split": sample_id.split("_")[0],
                    "metadata": {"dominant_transition": "1->2"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    write_manifest(train_manifest, "train_1", train_before, train_after)
    write_manifest(val_manifest, "val_1", val_before, val_after)
    write_manifest(test_manifest, "test_1", test_before, test_after)

    output_dir = tmp_path / "split_out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_semantic_change.py",
            "--train-manifest",
            str(train_manifest),
            "--val-manifest",
            str(val_manifest),
            "--test-manifest",
            str(test_manifest),
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

    assert result.returncode == 0, result.stderr
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["train_manifest"] == str(train_manifest)
    assert config["val_manifest"] == str(val_manifest)
    assert config["test_manifest"] == str(test_manifest)
    assert config["train_samples"] == 1
    assert config["val_samples"] == 1
    assert config["test_samples"] == 1
    assert (output_dir / "semantic_eval_test.json").exists()
