#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from land_change_detection.data.qcpr_dataset_v2 import (
    SCHEMA_VERSION,
    build_rows_v2,
    exclude_cross_split_image_conflicts,
    jsonl_read,
    jsonl_write,
    leakage_audit,
    mask_free,
    split_role,
    stable_manifest_hashes,
    validate_relevance,
)


def _read_sources(manifest_dir: Path) -> list[dict]:
    candidates = [
        (manifest_dir / "natural_train_retrieval_manifest.jsonl", "train"),
        (manifest_dir / "natural_validation_retrieval_manifest.jsonl", "development"),
        (manifest_dir / "natural_test_retrieval_manifest.jsonl", "test"),
    ]
    rows: list[dict] = []
    for path, fallback_split in candidates:
        for row in jsonl_read(path):
            item = dict(row)
            item["split"] = split_role(item.get("split") or fallback_split)
            rows.append(item)
    if not rows:
        raise RuntimeError(f"no source rows found in {manifest_dir}")
    return rows


def _task_row(pair: dict, caption: dict, relevance: dict, split: str) -> dict:
    return mask_free(
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_pair_id": pair["canonical_pair_id"],
            "caption_id": caption["caption_id"],
            "caption": caption["text"],
            "query_scope": caption["query_scope"],
            "semantic_group_id": caption.get("semantic_group_id"),
            "positive_pair_ids": relevance["positive_pair_ids"],
            "ignored_pair_ids": relevance["ignored_pair_ids"],
            "negative_policy": relevance["negative_policy"],
            "t1_path": pair.get("t1_path"),
            "t2_path": pair.get("t2_path"),
            "dataset_name": pair["source_dataset"],
            "split": split,
            "is_generated": caption.get("is_generated", False),
            "verification_status": caption.get("verification_status"),
            "change_status": caption.get("change_status"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_rows = _read_sources(args.manifest_dir)
    source_rows, split_resolution = exclude_cross_split_image_conflicts(source_rows)
    summary = build_rows_v2(source_rows, args.output_root / "registries")

    pairs = jsonl_read(args.output_root / "registries/pair_registry.jsonl")
    captions = jsonl_read(args.output_root / "registries/caption_registry.jsonl")
    relevance = jsonl_read(args.output_root / "registries/relevance_registry.jsonl")
    dense = jsonl_read(args.output_root / "registries/dense_label_registry.jsonl")

    pair_by_id = {item["canonical_pair_id"]: item for item in pairs}
    relevance_by_caption = {item["caption_id"]: item for item in relevance}

    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    leakage = leakage_audit(pairs)
    relevance_audit = validate_relevance(pairs, captions, relevance)
    (reports / "dataset_v2_leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n"
    )
    (reports / "dataset_v2_relevance_audit.json").write_text(
        json.dumps(relevance_audit, indent=2, sort_keys=True) + "\n"
    )

    scope_counts = Counter()
    split_counts = Counter()
    for split in ("train", "development", "test"):
        retrieval_rows: list[dict] = []
        semantic_rows: list[dict] = []
        grounding_rows: list[dict] = []
        for caption in captions:
            pair = pair_by_id[caption["canonical_pair_id"]]
            if pair["split"] != split:
                continue
            rel = relevance_by_caption[caption["caption_id"]]
            scope = caption["query_scope"]
            scope_counts[scope] += 1
            split_counts[split] += 1
            task = _task_row(pair, caption, rel, split)
            if scope in {"exact_pair", "semantic_group"}:
                retrieval_rows.append(task)
            if scope in {"semantic_group", "generic_no_change"}:
                semantic_rows.append(task)
            if scope in {"exact_pair", "semantic_group", "localized_query"}:
                grounding_rows.append(task)

        jsonl_write(args.output_root / f"retrieval_{split}_v2.jsonl", retrieval_rows)
        jsonl_write(args.output_root / f"retrieval_semantic_{split}_v2.jsonl", semantic_rows)
        jsonl_write(args.output_root / f"grounding_{split}_mask_free_v2.jsonl", grounding_rows)

    jsonl_write(args.output_root / "dense_evaluation_v2.jsonl", dense)
    jsonl_write(args.output_root / "scene_language_pretrain_v2.jsonl", [])

    summary.update(
        {
            "status": "DATASET_V2_CORE_READY" if leakage["passed"] and relevance_audit["passed"] else "DATA_QUALITY_HOLD",
            "split_conflict_resolution": split_resolution,
            "leakage_audit_passed": leakage["passed"],
            "relevance_audit_passed": relevance_audit["passed"],
            "query_scope_counts": dict(scope_counts),
            "task_row_counts_by_split": dict(split_counts),
            "manifest_hashes": stable_manifest_hashes(args.output_root),
        }
    )
    (args.output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not leakage["passed"]:
        raise RuntimeError("Dataset-v2 leakage audit failed")
    if not relevance_audit["passed"]:
        raise RuntimeError("Dataset-v2 relevance audit failed")


if __name__ == "__main__":
    main()
