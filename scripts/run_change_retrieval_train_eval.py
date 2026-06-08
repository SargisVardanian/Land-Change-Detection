from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight retrieval train/eval scaffold on a generated manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_output = args.output_dir / "eval_metrics.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_retrieval_head.py",
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--epochs",
            str(args.epochs),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_retrieval.py",
            "--manifest",
            str(args.manifest),
            "--output",
            str(eval_output),
        ],
        check=True,
    )
    print(f"Train/eval complete -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
