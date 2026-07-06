from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import audit_manifest_rows, load_json_rows, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge canonical temporal-caption manifests without re-splitting.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for manifest in args.manifest:
        rows.extend(load_json_rows(manifest))
    report = audit_manifest_rows(rows)
    if not report["valid"] and not args.allow_errors:
        raise SystemExit(f"Merged manifest audit failed: {report['errors'][:3]}")
    write_jsonl(args.output, rows)
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
