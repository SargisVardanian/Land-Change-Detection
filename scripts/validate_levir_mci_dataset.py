from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from land_change_detection.levir_mci import discover_levir_mci_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real LEVIR-MCI files and write a reusable index.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--index-output", type=Path, default=None)
    parser.add_argument("--check-all", action="store_true", help="Open all images instead of only the first 64 samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_levir_mci_samples(args.data_root)
    if not samples:
        raise SystemExit("No LEVIR-MCI samples discovered under the provided data root.")

    inspect_limit = len(samples) if args.check_all else min(len(samples), 64)
    mismatched_sizes: list[dict[str, object]] = []
    missing_captions = 0
    split_counts = Counter(sample.split for sample in samples)

    for sample in samples[:inspect_limit]:
        before_size = Image.open(sample.image_before).size
        after_size = Image.open(sample.image_after).size
        mask_size = Image.open(sample.binary_change_mask).size
        if before_size != after_size or before_size != mask_size:
            mismatched_sizes.append(
                {
                    "sample_id": sample.sample_id,
                    "before_size": before_size,
                    "after_size": after_size,
                    "mask_size": mask_size,
                }
            )
        if not sample.caption.strip():
            missing_captions += 1

    output_path = args.output or (args.project_root / "indexes" / "levir_mci_validation.json")
    index_output = args.index_output or (args.project_root / "indexes" / "levir_mci_samples.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.write_text(
        "\n".join(json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )

    report = {
        "data_root": str(args.data_root),
        "num_samples": len(samples),
        "split_counts": dict(split_counts),
        "checked_samples": inspect_limit,
        "missing_captions_in_checked_subset": missing_captions,
        "mismatched_sizes": mismatched_sizes,
        "index_output": str(index_output),
        "ready": len(samples) > 0 and not mismatched_sizes,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
