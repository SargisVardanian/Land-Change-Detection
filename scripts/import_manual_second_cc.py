from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.manual_dataset_imports import (
    build_grouped_semantic_samples,
    build_transition_manifest_rows,
    canonical_import_outputs,
    directory_inventory,
    ensure_symlink,
    samples_to_rows,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and import a manually downloaded SECOND-CC dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root or args.project_root / "datasets" / "manual" / "SECOND-CC"
    if not root.exists():
        raise SystemExit(f"Manual SECOND-CC root not found: {root}")
    samples = build_grouped_semantic_samples(root, "SECOND-CC")
    if not samples:
        raise SystemExit(f"No SECOND-CC before/after pairs discovered under {root}")
    outputs = canonical_import_outputs(args.project_root, "second_cc")
    sample_manifest = write_jsonl(outputs["sample_manifest"], samples_to_rows(samples))
    pair_rows = build_transition_manifest_rows(samples, dataset_name="SECOND-CC", num_classes=args.num_classes)
    if not pair_rows:
        raise SystemExit(f"No SECOND-CC semantic before/after labels discovered under {root}")
    pair_manifest = write_jsonl(outputs["pair_manifest"], pair_rows)
    raw_link = ensure_symlink(args.project_root / "datasets" / "raw" / "SECOND-CC", root)
    payload = {
        "dataset_name": "SECOND-CC",
        "manual_root": str(root),
        "raw_link": str(raw_link),
        "inventory": directory_inventory(root),
        "sample_manifest": str(sample_manifest),
        "pair_manifest": str(pair_manifest),
        "num_samples": len(samples),
        "num_pair_rows": len(pair_rows),
    }
    write_json(root / "IMPORT_PROVENANCE.json", payload)
    write_json(outputs["report"], payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
