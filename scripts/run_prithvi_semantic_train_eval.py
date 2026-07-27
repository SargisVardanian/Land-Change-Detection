from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six-band semantic baseline from a Prithvi-compatible manifest. "
            "This is not TerraTorch Prithvi foundation-model fine-tuning."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best.pt"
    eval_output = args.output_dir / "eval_metrics.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_semantic_change.py",
            "--manifest",
            str(args.manifest),
            "--train-manifest",
            str(args.manifest),
            "--val-manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--input-channels",
            "6",
            "--num-classes",
            str(args.num_classes),
            "--run-tag",
            "six_band_semantic_baseline",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_semantic_change.py",
            "--manifest",
            str(args.manifest),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(eval_output),
            "--batch-size",
            str(args.batch_size),
            "--input-channels",
            "6",
            "--num-classes",
            str(args.num_classes),
            "--run-tag",
            "six_band_semantic_baseline",
        ],
        check=True,
    )
    print(f"Six-band semantic baseline complete -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
