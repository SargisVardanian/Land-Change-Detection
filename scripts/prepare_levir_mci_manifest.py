from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.levir_mci import discover_levir_mci_samples
from land_change_detection.temporal_caption_manifest import audit_manifest_rows, make_manifest_row, write_jsonl


def build_rows(root: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    for sample in discover_levir_mci_samples(root):
        if split != "all" and sample.split != split:
            continue
        captions = [caption for caption in sample.captions if caption.strip()]
        if not captions and sample.caption.strip():
            captions = [sample.caption]
        rows.append(
            make_manifest_row(
                dataset_name="levir_mci",
                split=sample.split,
                original_id=sample.sample_id,
                t1_path=sample.image_before,
                t2_path=sample.image_after,
                captions=captions,
                caption_source="human",
                mask_path=sample.binary_change_mask,
                source_metadata={"dataset": "LEVIR-MCI", **sample.metadata},
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical temporal-caption manifest for LEVIR-MCI.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.root, args.split)
    write_jsonl(args.output, rows)
    report = audit_manifest_rows(rows)
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
