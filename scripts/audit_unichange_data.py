from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.preprocess_levir_cc import _build_records, _records_to_rows, _write_jsonl
from land_change_detection.run_metadata import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniChange data audit for LEVIR-CC and LEVIR-MCI manifests.")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--levir-cc-root", type=Path, default=REPO_ROOT / "datasets" / "raw" / "LEVIR-CC")
    parser.add_argument("--pair-output", type=Path, default=REPO_ROOT / "indexes" / "levir_cc_pairs.jsonl")
    parser.add_argument("--caption-output", type=Path, default=REPO_ROOT / "indexes" / "levir_cc_caption_queries.jsonl")
    parser.add_argument("--cc-report", type=Path, default=REPO_ROOT / "reports" / "data_audit" / "levir_cc.json")
    parser.add_argument("--mci-a", type=Path, default=REPO_ROOT / "indexes" / "levir_mci_samples.jsonl")
    parser.add_argument("--mci-b", type=Path, default=REPO_ROOT / "indexes" / "levir_mci_samples_fixed.jsonl")
    parser.add_argument("--mci-report", type=Path, default=REPO_ROOT / "reports" / "data_audit" / "levir_mci_manifest_diff.json")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def _field(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _manifest_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [_field(row, ("pair_id", "sample_id", "id")) for row in rows]
    splits = [_field(row, ("split", "partition", "subset")) for row in rows]
    path_fields = ("t1", "t2", "before_path", "after_path", "image_path", "mask_path", "change_mask_path")
    referenced_paths = sorted({str(row.get(key)) for row in rows for key in path_fields if row.get(key)})
    duplicates = [key for key, count in Counter(ids).items() if key and count > 1]
    split_by_id: dict[str, set[str]] = {}
    for row_id, split in zip(ids, splits, strict=False):
        if not row_id:
            continue
        split_by_id.setdefault(row_id, set()).add(split)
    leakage = sorted(row_id for row_id, values in split_by_id.items() if len(values - {""}) > 1)
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() else None,
        "row_count": len(rows),
        "unique_id_count": len({row_id for row_id in ids if row_id}),
        "duplicate_id_count": len(duplicates),
        "duplicate_id_preview": duplicates[:25],
        "split_counts": dict(Counter(split or "unknown" for split in splits)),
        "leakage_count": len(leakage),
        "leakage_preview": leakage[:25],
        "referenced_path_count": len(referenced_paths),
    }


def _compare_mci(args: argparse.Namespace) -> dict[str, Any]:
    rows_a = _read_jsonl(args.mci_a)
    rows_b = _read_jsonl(args.mci_b)
    norm_a = {json.dumps({k: v for k, v in row.items() if k != "_line_number"}, sort_keys=True): row for row in rows_a}
    norm_b = {json.dumps({k: v for k, v in row.items() if k != "_line_number"}, sort_keys=True): row for row in rows_b}
    ids_a = {_field(row, ("pair_id", "sample_id", "id")) for row in rows_a}
    ids_b = {_field(row, ("pair_id", "sample_id", "id")) for row in rows_b}
    return {
        "manifest_a": _manifest_summary(args.mci_a, rows_a),
        "manifest_b": _manifest_summary(args.mci_b, rows_b),
        "exact_row_intersection_count": len(set(norm_a) & set(norm_b)),
        "only_a_row_count": len(set(norm_a) - set(norm_b)),
        "only_b_row_count": len(set(norm_b) - set(norm_a)),
        "only_a_id_preview": sorted((ids_a - ids_b) - {""})[:25],
        "only_b_id_preview": sorted((ids_b - ids_a) - {""})[:25],
        "canonical_recommendation": "undecided_until_review" if rows_a and rows_b and set(norm_a) != set(norm_b) else "identical_or_missing",
    }


def main() -> int:
    args = parse_args()
    args.pair_output.parent.mkdir(parents=True, exist_ok=True)
    args.caption_output.parent.mkdir(parents=True, exist_ok=True)
    args.cc_report.parent.mkdir(parents=True, exist_ok=True)
    args.mci_report.parent.mkdir(parents=True, exist_ok=True)

    cc_report: dict[str, Any]
    try:
        records, source_report = _build_records(args.levir_cc_root, max_pairs=None)
        pair_rows, caption_rows = _records_to_rows(records, args.project_root)
        pair_rows = [
            {
                "pair_id": row["pair_id"],
                "t1": row["before_path"],
                "t2": row["after_path"],
                "captions": [caption["caption"] for caption in row.get("captions", [])],
                "split": row["split"],
            }
            for row in pair_rows
        ]
        _write_jsonl(args.pair_output, pair_rows)
        _write_jsonl(args.caption_output, caption_rows)
        pair_splits: dict[str, set[str]] = {}
        for row in pair_rows:
            pair_splits.setdefault(row["pair_id"], set()).add(row["split"])
        leakage = sorted(pair_id for pair_id, splits in pair_splits.items() if len(splits) > 1)
        missing_files = [
            path
            for row in pair_rows
            for path in (args.project_root / row["t1"], args.project_root / row["t2"])
            if not path.exists()
        ]
        if leakage:
            raise ValueError(f"Pair leakage detected: {leakage[:10]}")
        if missing_files:
            raise FileNotFoundError(f"Missing LEVIR-CC files: {missing_files[:10]}")
        cc_report = {
            "status": "ok",
            "pair_manifest": str(args.pair_output),
            "caption_query_manifest": str(args.caption_output),
            "pair_count": len(pair_rows),
            "caption_count": len(caption_rows),
            "pairs_by_split": dict(Counter(row["split"] for row in pair_rows)),
            "captions_by_split": dict(Counter(row["split"] for row in caption_rows)),
            "source_report": source_report,
        }
    except Exception as exc:
        cc_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "root": str(args.levir_cc_root)}
    args.cc_report.write_text(json.dumps(cc_report, indent=2, ensure_ascii=False), encoding="utf-8")

    mci_report = _compare_mci(args)
    args.mci_report.write_text(json.dumps(mci_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"levir_cc": cc_report["status"], "mci": mci_report["canonical_recommendation"]}, indent=2))
    return 0 if cc_report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
