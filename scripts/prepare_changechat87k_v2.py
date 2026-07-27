#!/usr/bin/env python3
"""Map official ChangeChat instructions to canonical LEVIR-MCI pairs.

The script never copies LEVIR imagery and never silently discards unmapped
records: a machine-readable mapping report is mandatory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from land_change_detection.data.qcpr_dataset_v2 import (
    SCHEMA_VERSION,
    jsonl_read,
    jsonl_write,
    normalize_text,
)


def _candidate_source_ids(row: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("pair_id", "image_id", "source_pair_id", "id", "image", "image_name"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        candidates.add(text)
        candidates.add(Path(text).stem)
    return candidates


def _task_scope(task_type: str) -> str:
    value = task_type.casefold()
    if "local" in value or "region" in value:
        return "localized_query"
    if "caption" in value:
        return "semantic_group"
    return "instruction_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--levir-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-mapping-rate", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pair_rows = jsonl_read(args.levir_pairs)
    canonical_ids = {str(row["canonical_pair_id"]) for row in pair_rows}
    index: dict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        canonical_id = str(row["canonical_pair_id"])
        source_id = str(row.get("source_pair_id") or "")
        for key in {canonical_id, source_id, Path(source_id).stem}:
            if key:
                index[key].add(canonical_id)

    instructions = jsonl_read(args.instructions)
    if args.limit > 0:
        instructions = instructions[: args.limit]

    output: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    task_counts = Counter()

    for ordinal, row in enumerate(instructions):
        matches: set[str] = set()
        for candidate in _candidate_source_ids(row):
            if candidate in canonical_ids:
                matches.add(candidate)
            matches.update(index.get(candidate, set()))
        if not matches:
            unmapped.append({"ordinal": ordinal, "source_ids": sorted(_candidate_source_ids(row))})
            continue
        if len(matches) != 1:
            ambiguous.append({"ordinal": ordinal, "matches": sorted(matches)})
            continue

        canonical_id = next(iter(matches))
        prompt = str(
            row.get("instruction")
            or row.get("question")
            or row.get("prompt")
            or row.get("query")
            or ""
        ).strip()
        response = str(row.get("response") or row.get("answer") or row.get("output") or "").strip()
        task_type = str(row.get("task_type") or row.get("type") or "instruction")
        scope = _task_scope(task_type)
        text = prompt if scope in {"localized_query", "instruction_only"} else response
        if not text:
            unmapped.append({"ordinal": ordinal, "reason": "empty_text", "canonical_pair_id": canonical_id})
            continue

        generated = bool(row.get("gpt") or row.get("gpt_assisted") or "gpt" in str(row.get("source", "")).casefold())
        task_counts[task_type] += 1
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "caption_id": f"changechat:{canonical_id}:{ordinal}",
                "canonical_pair_id": canonical_id,
                "text": text,
                "normalized_text": normalize_text(text),
                "instruction": prompt,
                "response": response,
                "caption_source": "changechat_gpt_assisted" if generated else "changechat_rule_based",
                "task_type": task_type,
                "query_scope": scope,
                "semantic_group_id": f"changechat:{task_type.casefold().replace(' ', '_')}",
                "equivalent_caption_group_id": None,
                "quality_score": 0.0 if generated else 0.5,
                "identifiability_score": 0.0,
                "is_generated": generated,
                "generator": row.get("generator") or ("GPT-assisted" if generated else None),
                "verification_status": "generated_unverified" if generated else "rule_based_unverified",
                "conversation_id": row.get("conversation_id") or row.get("conversation") or row.get("dialogue_id"),
                "source_ordinal": ordinal,
                "retrieval_supervision": scope == "semantic_group",
            }
        )

    total = len(instructions)
    mapping_rate = len(output) / total if total else 0.0
    report = {
        "schema_version": SCHEMA_VERSION,
        "input_records": total,
        "mapped_records": len(output),
        "unmapped_records": len(unmapped),
        "ambiguous_records": len(ambiguous),
        "mapping_rate": mapping_rate,
        "task_counts": dict(task_counts),
        "unmapped_sample": unmapped[:100],
        "ambiguous_sample": ambiguous[:100],
        "passed": mapping_rate >= args.min_mapping_rate and not ambiguous,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    jsonl_write(args.output, output)
    if not report["passed"]:
        raise RuntimeError(
            f"ChangeChat canonical mapping failed: rate={mapping_rate:.3f}, ambiguous={len(ambiguous)}"
        )


if __name__ == "__main__":
    main()
