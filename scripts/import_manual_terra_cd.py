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
    parser = argparse.ArgumentParser(description="Validate and import a manually downloaded TERRA-CD dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root or args.project_root / "datasets" / "manual" / "TERRA-CD"
    if not root.exists():
        raise SystemExit(f"Manual TERRA-CD root not found: {root}")
    samples = build_grouped_semantic_samples(root, "TERRA-CD")
    if not samples:
        note = root / "MANUAL_DOWNLOAD_REQUIRED.md"
        note.write_text(
            "No canonical TERRA-CD before/after pairs were discovered here.\n"
            "Check the archive layout and place the official dataset contents under this directory.\n",
            encoding="utf-8",
        )
        raise SystemExit(f"No TERRA-CD before/after pairs discovered under {root}")
    outputs = canonical_import_outputs(args.project_root, "terra_cd")
    sample_manifest = write_jsonl(outputs["sample_manifest"], samples_to_rows(samples))
    pair_manifest = None
    if args.num_classes is not None:
        pair_rows = build_transition_manifest_rows(samples, dataset_name="TERRA-CD", num_classes=args.num_classes)
        if pair_rows:
            pair_manifest = write_jsonl(outputs["pair_manifest"], pair_rows)
    raw_link = ensure_symlink(args.project_root / "datasets" / "raw" / "TERRA-CD", root)
    payload = {
        "dataset_name": "TERRA-CD",
        "manual_root": str(root),
        "raw_link": str(raw_link),
        "inventory": directory_inventory(root),
        "sample_manifest": str(sample_manifest),
        "pair_manifest": str(pair_manifest) if pair_manifest else None,
        "num_samples": len(samples),
    }
    write_json(root / "IMPORT_PROVENANCE.json", payload)
    write_json(outputs["report"], payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
