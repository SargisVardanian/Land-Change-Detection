#!/usr/bin/env python3
"""Profile an immutable Dataset-v2 release and emit inspectable quality reports."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN = {"mask", "masks", "semantic", "target", "targets", "label", "labels", "ground_truth", "gt", "query_masks"}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.exists() else []


def forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if str(key).casefold().split("_")[-1] in FORBIDDEN or str(key).casefold() in FORBIDDEN:
                found.append(path)
            found.extend(forbidden_keys(nested, path))
    elif isinstance(value, list):
        for i, nested in enumerate(value):
            found.extend(forbidden_keys(nested, f"{prefix}[{i}]"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reg = args.release / "registries"
    pairs = rows(reg / "pair_registry.jsonl")
    captions = rows(reg / "caption_registry.jsonl")
    instructions = rows(reg / "instruction_registry.jsonl")
    dense = rows(reg / "dense_label_registry.jsonl")
    all_text = captions + instructions
    by_source = Counter(str(row.get("source_dataset") or row.get("dataset_name") or "unknown") for row in pairs)
    def retrieval_flag(row: dict[str, Any]) -> bool:
        value = row.get("retrieval_supervision")
        return bool(value) if value is not None else str(row.get("query_scope")) in {"exact_pair", "semantic_group"}

    def grounding_flag(row: dict[str, Any]) -> bool:
        value = row.get("grounding_supervision")
        return bool(value) if value is not None else str(row.get("query_scope")) in {"localized_query", "semantic_group"} and not retrieval_flag(row)

    retrieval_flags = [retrieval_flag(row) for row in all_text]
    grounding_flags = [grounding_flag(row) for row in all_text]
    composition = {
        "release": str(args.release),
        "physical_pairs": len(pairs),
        "pair_counts_by_source": dict(sorted(by_source.items())),
        "caption_records": len(captions),
        "instruction_records": len(instructions),
        "text_records": len(all_text),
        "dense_label_records": len(dense),
        "caption_source": dict(Counter(str(row.get("caption_source")) for row in all_text)),
        "verification_status": dict(Counter(str(row.get("verification_status")) for row in all_text)),
        "query_scope": dict(Counter(str(row.get("query_scope")) for row in all_text)),
        "retrieval_supervised": sum(retrieval_flags),
        "grounding_supervised": sum(grounding_flags),
        "grounding_only": sum(not r and g for r, g in zip(retrieval_flags, grounding_flags)),
        "excluded_from_training": sum(not r and not g for r, g in zip(retrieval_flags, grounding_flags)),
        "generated": sum(bool(row.get("is_generated")) for row in all_text),
        "human": sum(str(row.get("caption_source")) == "human" for row in all_text),
        "human_temporal_captions": sum(str(row.get("caption_source")) == "human" for row in captions),
        "verified_generated_temporal_captions": sum(bool(row.get("is_generated")) and str(row.get("verification_status", "")).casefold() in {"verified", "human_verified", "accepted"} for row in captions),
        "unverified_generated_temporal_captions": sum(bool(row.get("is_generated")) and str(row.get("verification_status", "")).casefold() not in {"verified", "human_verified", "accepted"} for row in captions),
        "exact_pair_queries": sum(str(row.get("query_scope")) == "exact_pair" for row in all_text),
        "semantic_group_queries": sum(str(row.get("query_scope")) == "semantic_group" for row in all_text),
        "generic_no_change_queries": sum(str(row.get("query_scope")) == "generic_no_change" for row in all_text),
        "localized_queries": sum(str(row.get("query_scope")) == "localized_query" for row in all_text),
        "instruction_only_records": sum(str(row.get("query_scope")) == "instruction_only" for row in all_text),
        "identifiability": {
            "min": min((float(row.get("identifiability_score") or 0.0) for row in all_text), default=0.0),
            "median": statistics.median((float(row.get("identifiability_score") or 0.0) for row in all_text)) if all_text else 0.0,
            "max": max((float(row.get("identifiability_score") or 0.0) for row in all_text), default=0.0),
        },
    }
    ids = [str(row.get("canonical_pair_id")) for row in pairs]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    relevance_overlap: list[str] = []
    for row in rows(reg / "relevance_registry.jsonl"):
        positives = set(row.get("positive_pair_ids") or [])
        ignored = set(row.get("ignored_pair_ids") or [])
        if positives & ignored:
            relevance_overlap.append(str(row.get("caption_id")))
    mask_free_violations: list[dict[str, Any]] = []
    for path in sorted((args.release / "manifests").glob("*.jsonl")):
        if "grounding_mask_free" not in path.name and "retrieval" not in path.name:
            continue
        for index, row in enumerate(rows(path)):
            keys = forbidden_keys(row)
            if keys:
                mask_free_violations.append({"path": str(path), "line": index + 1, "keys": keys[:20]})
    integrity = {
        "passed": not duplicate_ids and not relevance_overlap and not mask_free_violations,
        "duplicate_canonical_ids": duplicate_ids,
        "positive_ignored_overlap": relevance_overlap,
        "mask_free_violations": mask_free_violations[:100],
        "mask_free_violation_count": len(mask_free_violations),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "current_release_composition.json").write_text(json.dumps(composition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "current_release_integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Current Dataset-v2 Expanded composition", "", f"Release: `{args.release}`", "", "## Counts", "", "| Field | Count |", "|---|---:|"]
    for key in ("physical_pairs", "caption_records", "instruction_records", "text_records", "dense_label_records", "retrieval_supervised", "grounding_supervised", "grounding_only", "excluded_from_training", "generated", "human"):
        md.append(f"| {key} | {composition[key]} |")
    md += ["", "## By source", "", "```json", json.dumps(composition["pair_counts_by_source"], indent=2), "```", "", "## Quality gates", "", f"Integrity passed: **{integrity['passed']}**", "", "The release is profiled without changing its files."]
    (args.output / "current_release_composition.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0 if integrity["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
