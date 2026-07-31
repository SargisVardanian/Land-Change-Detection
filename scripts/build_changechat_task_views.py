#!/usr/bin/env python3
"""Route mapped ChangeChat records into bounded task-specific views.

This is a data-view operation: it never copies imagery and never changes the
canonical physical-pair mapping or inherited split.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TASKS = ("captioning", "detection", "counting", "localized", "multiturn")


def classify(row: dict) -> set[str]:
    text = f"{row.get('instruction', '')} {row.get('response', '')}".casefold()
    original = row.get("original_record", {})
    conversations = original.get("conversations", [])
    task = str(row.get("task_type", "")).casefold()
    out: set[str] = set()
    if "caption" in task or any(x in text for x in ("describe", "what changed", "briefly describe")):
        out.add("captioning")
    if any(x in task or x in text for x in ("detect", "detection", "change", "difference", "yes or no", "binary")):
        out.add("detection")
    if any(x in task or x in text for x in ("count", "how many", "number of")):
        out.add("counting")
    if any(x in task or x in text for x in ("where", "location", "locate", "localiz", "region", "area")):
        out.add("localized")
    if len(conversations) > 2 or "multi" in task or "conversation" in task:
        out.add("multiturn")
    return out or {"detection"}


def normalize(row: dict, task: str, ordinal: int) -> dict:
    original = row.get("original_record", {})
    conversations = original.get("conversations", [])
    return {
        "record_id": row.get("instruction_id", f"changechat:{ordinal}"),
        "canonical_pair_id": row["canonical_pair_id"],
        "instruction": row.get("instruction", ""),
        "response": row.get("response", ""),
        "conversation_id": original.get("id", row.get("instruction_id")),
        "turn_index": max(0, len(conversations) - 2),
        "task_type": task,
        "source_task_type": row.get("task_type"),
        "rule_based_or_gpt": row.get("rule_based_or_gpt", row.get("verification_status", "unknown")),
        "verification_status": row.get("verification_status", "unknown"),
        "inherited_physical_split": row.get("split"),
        "is_instruction_only": task != "captioning",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--max-per-pair", type=int, default=4)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ordinal, line in enumerate(args.input.open(), 1):
        row = json.loads(line)
        for task in classify(row):
            buckets[task].append(normalize(row, task, ordinal))
    counts = {}
    for task in TASKS:
        rows = sorted(buckets[task], key=lambda x: (x["canonical_pair_id"], x["record_id"]))
        kept: list[dict] = []
        per_pair: dict[str, int] = defaultdict(int)
        for row in rows:
            key = row["canonical_pair_id"]
            if per_pair[key] >= args.max_per_pair:
                continue
            per_pair[key] += 1
            kept.append(row)
        path = args.output_dir / f"changechat_{task}_train.jsonl"
        with path.open("w") as fh:
            for row in kept:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        counts[task] = {"input_records": len(rows), "kept_records": len(kept), "physical_pairs": len(per_pair)}
    (args.output_dir / "changechat_task_view_summary.json").write_text(json.dumps({"max_per_pair": args.max_per_pair, "counts": counts}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
