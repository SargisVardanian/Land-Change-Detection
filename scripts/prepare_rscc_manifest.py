from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import annotation_rows_to_manifest, audit_manifest_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an optional RSCC temporal-caption manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-model-generated", action="store_true")
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = annotation_rows_to_manifest(
        dataset_name="rscc",
        root=args.root,
        annotations=args.annotations,
        default_caption_source="human",
        include_model_generated=args.include_model_generated,
    )
    for row in rows:
        row.setdefault("source_metadata", {})["rscc_default_excludes_model_generated"] = not args.include_model_generated
        row["source_metadata"].setdefault("license_family", "xBD")
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
