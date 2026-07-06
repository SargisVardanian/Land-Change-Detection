from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.temporal_caption_manifest import audit_manifest_rows, load_json_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical temporal-caption manifests.")
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for manifest in args.manifest:
        rows.extend(load_json_rows(manifest))
    report = audit_manifest_rows(rows)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0 if report["valid"] or args.allow_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
