from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _seed_pair(path_before: Path, path_after: Path, before_color: tuple[int, int, int], after_color: tuple[int, int, int]) -> None:
    _write_rgb(path_before, before_color)
    _write_rgb(path_after, after_color)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_train_eval_query_pair_retrieval_cli(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    _seed_pair(data_root / "levir_a_before.png", data_root / "levir_a_after.png", (255, 0, 0), (0, 255, 0))
    _seed_pair(data_root / "levir_b_before.png", data_root / "levir_b_after.png", (250, 10, 10), (10, 250, 10))
    _seed_pair(data_root / "pair_c_before.png", data_root / "pair_c_after.png", (0, 0, 255), (255, 255, 0))
    _seed_pair(data_root / "pair_d_before.png", data_root / "pair_d_after.png", (0, 10, 250), (250, 250, 20))

    levir_manifest = tmp_path / "levir.jsonl"
    pair_manifest = tmp_path / "pair.jsonl"
    _write_jsonl(
        levir_manifest,
        [
            {
                "sample_id": "levir_a",
                "before_path": str(data_root / "levir_a_before.png"),
                "after_path": str(data_root / "levir_a_after.png"),
                "caption": "new building appeared near the road",
                "metadata": {"transition_label": "urban_growth"},
            },
            {
                "sample_id": "levir_b",
                "before_path": str(data_root / "levir_b_before.png"),
                "after_path": str(data_root / "levir_b_after.png"),
                "caption": "another new building appeared near the road",
                "metadata": {"transition_label": "urban_growth"},
            },
        ],
    )
    _write_jsonl(
        pair_manifest,
        [
            {
                "sample_id": "pair_c",
                "dataset_name": "SECOND-CC",
                "before_path": str(data_root / "pair_c_before.png"),
                "after_path": str(data_root / "pair_c_after.png"),
                "transition_histogram": [0.8, 0.2, 0.0, 0.0],
                "dominant_transition": "1->2",
            },
            {
                "sample_id": "pair_d",
                "dataset_name": "SECOND-CC",
                "before_path": str(data_root / "pair_d_before.png"),
                "after_path": str(data_root / "pair_d_after.png"),
                "transition_histogram": [0.75, 0.25, 0.0, 0.0],
                "dominant_transition": "1->2",
            },
        ],
    )

    output_dir = tmp_path / "run"
    env = {"PYTHONPATH": "src"}
    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_dino_pair_retrieval.py",
            "--levir-manifest",
            str(levir_manifest),
            "--pair-manifest",
            str(pair_manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--visual-backbone",
            "simple_patch",
            "--text-backbone",
            "simple_text",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert train.returncode == 0, train.stderr
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "metrics_history.json").exists()

    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/eval_dino_pair_retrieval.py",
            "--levir-manifest",
            str(levir_manifest),
            "--pair-manifest",
            str(pair_manifest),
            "--checkpoint",
            str(output_dir / "best.pt"),
            "--output",
            str(output_dir / "eval_metrics.json"),
            "--image-size",
            "32",
            "--visual-backbone",
            "simple_patch",
            "--text-backbone",
            "simple_text",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert "recall@5" in metrics
    assert "mAP" in metrics
    assert "median_rank" in metrics
    assert "transition_recall@5" in metrics
    assert "transition_top1_hit_rate" in metrics
    assert "transition_nDCG@10" in metrics
    assert "directionality_accuracy" in metrics
    assert "anchors_with_positive_ratio" in metrics
    assert "overall" in metrics
    assert "by_source" in metrics
    assert "dataset_summary" in metrics
    assert "SECOND-CC" in metrics["by_source"]

    query = subprocess.run(
        [
            sys.executable,
            "scripts/query_pair_to_pair_retrieval.py",
            "--manifest",
            str(pair_manifest),
            "--checkpoint",
            str(output_dir / "best.pt"),
            "--query-sample-id",
            "pair_c",
            "--output",
            str(output_dir / "query_topk.png"),
            "--image-size",
            "32",
            "--visual-backbone",
            "simple_patch",
            "--text-backbone",
            "simple_text",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert query.returncode == 0, query.stderr
    assert (output_dir / "query_topk.png").exists()


def test_train_cli_fails_cleanly_when_dinov2_path_is_missing(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    pair_manifest = tmp_path / "pair.jsonl"
    data_root = tmp_path / "data"
    _seed_pair(data_root / "pair_before.png", data_root / "pair_after.png", (0, 0, 255), (255, 255, 0))
    _write_jsonl(
        pair_manifest,
        [
            {
                "sample_id": "pair_a",
                "dataset_name": "SECOND-CC",
                "before_path": str(data_root / "pair_before.png"),
                "after_path": str(data_root / "pair_after.png"),
                "transition_histogram": [1.0, 0.0, 0.0, 0.0],
                "dominant_transition": "1->2",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_dino_pair_retrieval.py",
            "--pair-manifest",
            str(pair_manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--image-size",
            "32",
            "--visual-backbone",
            "dinov2",
            "--text-backbone",
            "simple_text",
            "--dinov2-model-path",
            str(tmp_path / "missing-dinov2-small"),
            "--local-files-only",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src"},
    )
    assert result.returncode != 0
    assert "DINOv2 model path not found" in result.stderr
    assert "simple_patch" in result.stderr
