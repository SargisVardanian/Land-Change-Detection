#!/usr/bin/env python3
"""Build an expanded Dataset-v2 draft without overstating source readiness."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from land_change_detection.data.qcpr_dataset_v2 import (
    SCHEMA_VERSION,
    build_relevance,
    jsonl_read,
    jsonl_write,
    leakage_audit,
    legacy_pair_to_v2,
    mask_free,
    normalize_text,
    split_role,
    stable_manifest_hashes,
    validate_relevance,
)


def _s2_records(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    pairs: list[dict] = []
    captions: list[dict] = []
    dense: list[dict] = []
    for row in jsonl_read(path):
        legacy = {
            "pair_id": row["pair_id"],
            "dataset_name": "S2Looking",
            "split": split_role(row.get("split", "unknown")),
            "t1_path": row.get("t1_path"),
            "t2_path": row.get("t2_path"),
            "source_metadata": row.get("source_metadata")
            or {"scene_id": row.get("original_id", row["pair_id"])},
            "license": row.get("license") or "S2Looking research terms",
        }
        pair = legacy_pair_to_v2(legacy, dataset_version="s2looking_manifest")
        pair["retrieval_supervision"] = False
        pairs.append(pair)

        query_masks: dict[str, str] = {}
        for index, target in enumerate(row.get("directional_targets") or []):
            text = str(target.get("caption") or "").strip()
            if not text:
                continue
            caption_id = f"{pair['canonical_pair_id']}:localized:{index}"
            captions.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "caption_id": caption_id,
                    "canonical_pair_id": pair["canonical_pair_id"],
                    "text": text,
                    "normalized_text": normalize_text(text),
                    "caption_source": row.get("caption_source") or "derived_unverified",
                    "task_type": "localized_grounding",
                    "query_scope": "localized_query",
                    "semantic_group_id": f"{pair['canonical_pair_id']}:direction:{target.get('direction', index)}",
                    "equivalent_caption_group_id": None,
                    "quality_score": float(target.get("caption_confidence", 0.0)),
                    "identifiability_score": 0.0,
                    "verification_status": "derived_not_human_reviewed",
                    "is_generated": True,
                    "generator": "S2Looking label-derived query builder",
                    "change_status": "changed",
                    "dataset_name": "S2Looking",
                    "retrieval_supervision": False,
                }
            )
            if target.get("mask_path"):
                query_masks[caption_id] = str(target["mask_path"])
        if query_masks:
            dense.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "canonical_pair_id": pair["canonical_pair_id"],
                    "query_masks": query_masks,
                    "label_source": "S2Looking",
                }
            )
    return pairs, captions, dense


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
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--s2-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    core_pairs = jsonl_read(args.core_root / "registries/pair_registry.jsonl")
    core_captions = jsonl_read(args.core_root / "registries/caption_registry.jsonl")
    core_relevance = jsonl_read(args.core_root / "registries/relevance_registry.jsonl")
    core_dense = jsonl_read(args.core_root / "registries/dense_label_registry.jsonl")
    if not core_pairs or not core_captions:
        raise RuntimeError("core Dataset-v2 registries are empty")

    s2_pairs, s2_captions, s2_dense = _s2_records(args.s2_manifest)
    pairs = core_pairs + s2_pairs
    captions = core_captions + s2_captions
    relevance = core_relevance + build_relevance(s2_captions)
    dense = core_dense + s2_dense

    pair_ids = [item["canonical_pair_id"] for item in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("expanded registry contains duplicate canonical_pair_id values")

    out = args.output_root
    registries = out / "registries"
    reports = out / "reports"
    registries.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    jsonl_write(registries / "pair_registry.jsonl", pairs)
    jsonl_write(registries / "caption_registry.jsonl", captions)
    jsonl_write(registries / "relevance_registry.jsonl", relevance)
    jsonl_write(registries / "dense_label_registry.jsonl", dense)

    pair_by_id = {item["canonical_pair_id"]: item for item in pairs}
    rel_by_caption = {item["caption_id"]: item for item in relevance}
    counts = Counter()

    for split in ("train", "development", "test"):
        retrieval_rows: list[dict] = []
        grounding_rows: list[dict] = []
        for caption in captions:
            pair = pair_by_id[caption["canonical_pair_id"]]
            if split_role(pair["split"]) != split:
                continue
            relevance_row = rel_by_caption[caption["caption_id"]]
            task = _task_row(pair, caption, relevance_row, split)
            if caption.get("retrieval_supervision", True) and caption["query_scope"] in {
                "exact_pair",
                "semantic_group",
            }:
                retrieval_rows.append(task)
                counts[f"retrieval:{split}:{pair['source_dataset']}"] += 1
            if caption["query_scope"] in {"exact_pair", "semantic_group", "localized_query"}:
                grounding_rows.append(task)
                counts[f"grounding:{split}:{pair['source_dataset']}"] += 1
        jsonl_write(out / f"retrieval_{split}_v2_expanded.jsonl", retrieval_rows)
        jsonl_write(out / f"grounding_{split}_mask_free_v2_expanded.jsonl", grounding_rows)

    jsonl_write(out / "dense_evaluation_v2_expanded.jsonl", dense)
    jsonl_write(out / "scene_language_pretrain_v2_expanded.jsonl", [])

    leakage = leakage_audit(pairs)
    relevance_audit = validate_relevance(pairs, captions, relevance)
    (reports / "dataset_v2_expanded_leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n"
    )
    (reports / "dataset_v2_expanded_relevance_audit.json").write_text(
        json.dumps(relevance_audit, indent=2, sort_keys=True) + "\n"
    )

    blockers = [
        "ChangeChat official instruction artifact and canonical mapping",
        "Synthetic RCD SECOND original-A mapping",
        "SYSU-CD acquisition and caption-verification pilot",
        "Hi-UCD corrected release and pilot",
        "RSCC accessible assets and event-level split audit",
        "auxiliary scene-language datasets",
        "controlled old-data versus expanded-data pilot",
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXPANDED_DRAFT_NOT_READY",
        "sources_included": ["LEVIR-MCI", "SECOND-CC", "S2Looking_grounding_dense"],
        "sources_blocked": blockers,
        "pairs": len(pairs),
        "captions": len(captions),
        "dense_labels": len(dense),
        "task_counts": dict(counts),
        "leakage_audit_passed": leakage["passed"],
        "relevance_audit_passed": relevance_audit["passed"],
        "manifest_hashes": stable_manifest_hashes(out),
        "registry_hashes": stable_manifest_hashes(registries),
    }
    (reports / "dataset_v2_expanded_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not leakage["passed"]:
        raise RuntimeError("expanded Dataset-v2 leakage audit failed")
    if not relevance_audit["passed"]:
        raise RuntimeError("expanded Dataset-v2 relevance audit failed")


if __name__ == "__main__":
    main()
