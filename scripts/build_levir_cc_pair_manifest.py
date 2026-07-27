from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_levir_cc import _build_records, _overfit_rows, _records_to_rows, _split_rows


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
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Optional explicit directory for split-specific caption-query views.",
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
    split_output_dir = args.split_output_dir.resolve() if args.split_output_dir is not None else caption_output.parent
    train_output = split_output_dir / "levir_cc_caption_queries_train.jsonl"
    val_output = split_output_dir / "levir_cc_caption_queries_val.jsonl"
    test_output = split_output_dir / "levir_cc_caption_queries_test.jsonl"
    overfit_output = split_output_dir / "levir_cc_caption_queries_overfit_100.jsonl"

    train_rows = _split_rows(caption_rows, "train")
    val_rows = _split_rows(caption_rows, "val")
    test_rows = _split_rows(caption_rows, "test")
    overfit_rows = _overfit_rows(caption_rows)

    _write_jsonl(pair_output, pair_rows)
    _write_jsonl(caption_output, caption_rows)
    _write_jsonl(train_output, train_rows)
    _write_jsonl(val_output, val_rows)
    _write_jsonl(test_output, test_rows)
    _write_jsonl(overfit_output, overfit_rows)
    rendered = {
        "pair_manifest": str(pair_output),
        "caption_query_manifest": str(caption_output),
        "train_caption_query_manifest": str(train_output),
        "val_caption_query_manifest": str(val_output),
        "test_caption_query_manifest": str(test_output),
        "overfit_caption_query_manifest": str(overfit_output),
        "unique_pair_count": len(pair_rows),
        "caption_row_count": len(caption_rows),
        "split_source_counts": {
            source: sum(1 for row in pair_rows if row.get("split_source") == source)
            for source in sorted({str(row.get("split_source") or "unknown") for row in pair_rows})
        },
        "fallback_pair_count": sum(1 for row in pair_rows if row.get("split_source") == "fallback_deterministic_non_official"),
        "split_pair_counts": {
            split: len({row["pair_id"] for row in caption_rows if row["split"] == split})
            for split in sorted({row["split"] for row in caption_rows})
        },
        "split_caption_row_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "overfit_100": len(overfit_rows),
        },
    }
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
