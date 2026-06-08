from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.change_retrieval_datasets import discover_change_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index LEVIR-MCI or SECOND-CC style folders into a JSONL sample list.")
    parser.add_argument("--dataset-name", choices=("LEVIR-MCI", "SECOND-CC"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_change_samples(args.root, args.dataset_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples),
        encoding="utf-8",
    )
    print(f"Indexed {len(samples)} samples from {args.dataset_name} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
