from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from land_change_detection.levir_mci import discover_levir_mci_samples
from land_change_detection.temporal_caption_manifest import audit_manifest_rows, file_fingerprint, make_manifest_row


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


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


def apply_cross_split_duplicate_policy(rows: list[dict], policy: str) -> tuple[list[dict], dict[str, Any]]:
    by_pair_hash: defaultdict[tuple[str | None, str | None], list[dict]] = defaultdict(list)
    for row in rows:
        pair_hash = (file_fingerprint(row.get("t1_path")), file_fingerprint(row.get("t2_path")))
        by_pair_hash[pair_hash].append(row)
    removed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    remove_ids: set[str] = set()
    split_priority = {"test": 3, "val": 2, "train": 1}
    for pair_hash, group in by_pair_hash.items():
        splits = {str(row.get("split")) for row in group}
        if len(splits) <= 1:
            continue
        sorted_group = sorted(group, key=lambda row: (-split_priority.get(str(row.get("split")), 0), str(row.get("pair_id"))))
        retained = sorted_group[0]
        conflict = {
            "t1_sha": pair_hash[0],
            "t2_sha": pair_hash[1],
            "splits": sorted(splits),
            "retained_pair_id": retained.get("pair_id"),
            "retained_split": retained.get("split"),
            "conflicting_pair_ids": [row.get("pair_id") for row in sorted_group],
        }
        if policy == "error":
            errors.append(conflict)
            continue
        if str(retained.get("split")) not in {"test", "val"}:
            errors.append(conflict | {"reason": "drop_train can only resolve conflicts where test or val is retained"})
            continue
        for row in sorted_group[1:]:
            if row.get("split") == "train":
                remove_ids.add(str(row.get("pair_id")))
                removed.append(
                    {
                        "removed_pair_id": row.get("pair_id"),
                        "removed_split": row.get("split"),
                        "retained_pair_id": retained.get("pair_id"),
                        "retained_split": retained.get("split"),
                        "t1_sha": pair_hash[0],
                        "t2_sha": pair_hash[1],
                    }
                )
            else:
                errors.append(conflict | {"reason": "drop_train cannot remove non-train duplicates"})
    report = {
        "cross_split_duplicate_policy": policy,
        "removed_duplicates": removed,
        "unresolved_duplicates": errors,
    }
    if errors:
        raise ValueError(f"LEVIR-MCI cross-split duplicate leakage: {errors[:10]}")
    return [row for row in rows if str(row.get("pair_id")) not in remove_ids], report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical temporal-caption manifest for LEVIR-MCI.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--cross-split-duplicate-policy", choices=("error", "drop_train"), default="error")
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.root, args.split)
    try:
        rows, duplicate_report = apply_cross_split_duplicate_policy(rows, args.cross_split_duplicate_policy)
    except ValueError as exc:
        if args.audit_report:
            args.audit_report.parent.mkdir(parents=True, exist_ok=True)
            args.audit_report.write_text(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(str(exc)) from exc
    report = audit_manifest_rows(rows)
    report = {**report, "levir_mci_adapter": duplicate_report}
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"Manifest audit failed: {report['errors'][:3]}")
    _write_jsonl_atomic(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
