from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a LEVIR-CC pair manifest for pair-level split integrity.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(rows: list[dict[str, Any]], manifest: Path) -> dict[str, Any]:
    pair_to_splits: dict[str, set[str]] = defaultdict(set)
    pair_caption_counts: Counter[str] = Counter()
    missing_caption_rows = 0
    missing_path_rows = 0
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("sample_id") or "")
        split = str(row.get("split") or "unknown")
        pair_to_splits[pair_id].add(split)
        pair_caption_counts[pair_id] += 1
        if not str(row.get("caption") or "").strip():
            missing_caption_rows += 1
        before_path = Path(str(row.get("before_path") or ""))
        after_path = Path(str(row.get("after_path") or ""))
        if not before_path.exists() or not after_path.exists():
            missing_path_rows += 1
    leaking_pair_ids = sorted(
        pair_id for pair_id, splits in pair_to_splits.items() if len({split for split in splits if split != "unknown"}) > 1
    )
    split_pair_counts = Counter()
    for pair_id, splits in pair_to_splits.items():
        known = sorted(split for split in splits if split != "unknown")
        split_pair_counts[known[0] if known else "unknown"] += 1
    report = {
        "manifest": str(manifest),
        "caption_row_count": len(rows),
        "unique_pair_count": len(pair_to_splits),
        "split_pair_counts": dict(split_pair_counts),
        "max_captions_per_pair": max(pair_caption_counts.values(), default=0),
        "min_captions_per_pair": min(pair_caption_counts.values(), default=0),
        "missing_caption_rows": missing_caption_rows,
        "missing_path_rows": missing_path_rows,
        "leaking_pair_ids": leaking_pair_ids,
        "valid": not leaking_pair_ids and missing_path_rows == 0,
    }
    return report


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.manifest)
    report = build_report(rows, args.manifest)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["valid"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
