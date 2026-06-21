from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_levir_cc import _build_records, _records_to_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coordinated LEVIR-CC pair and caption-query manifests.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output path for the pair-centric manifest.")
    parser.add_argument(
        "--caption-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the flattened caption-query manifest.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _absolutize_paths(rows: list[dict], records: list) -> list[dict]:
    absolute = {record.pair_id: (str(record.before_path), str(record.after_path)) for record in records}
    normalized: list[dict] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        before_path, after_path = absolute[pair_id]
        normalized.append({**row, "before_path": before_path, "after_path": after_path})
    return normalized


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve() if args.project_root is not None else None
    pair_records, report = _build_records(args.root.resolve(), None)
    if report["missing_files"] or report["corrupt_files"] or report["dimension_mismatches"] or report["split_leakage"]:
        raise SystemExit(json.dumps(report, indent=2))

    pair_rows, caption_rows = _records_to_rows(pair_records, project_root or args.root.resolve())
    if project_root is None:
        pair_rows = _absolutize_paths(pair_rows, pair_records)
        caption_rows = _absolutize_paths(caption_rows, pair_records)
    pair_output = args.output.resolve()
    caption_output = (
        args.caption_output.resolve()
        if args.caption_output is not None
        else pair_output.with_name("levir_cc_caption_queries.jsonl")
    )
    _write_jsonl(pair_output, pair_rows)
    _write_jsonl(caption_output, caption_rows)
    rendered = {
        "pair_manifest": str(pair_output),
        "caption_query_manifest": str(caption_output),
        "unique_pair_count": len(pair_rows),
        "caption_row_count": len(caption_rows),
        "split_pair_counts": {
            split: len({row["pair_id"] for row in caption_rows if row["split"] == split})
            for split in sorted({row["split"] for row in caption_rows})
        },
    }
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
