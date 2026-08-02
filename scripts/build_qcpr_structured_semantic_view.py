#!/usr/bin/env python3
"""Build a compact mask-free semantic view from official S2Looking labels.

Only the official coarse directional relation is promoted. Fine count/location
captions and empty directional masks are excluded. Dense labels remain a
sidecar joined by canonical_pair_id and are never emitted in the view.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


RELATIONS = {"appeared": "new buildings appeared", "disappeared": "buildings were demolished"}
EXPECTED = {"official_source_order": "Image2_to_Image1", "label1": "appeared", "label2": "disappeared"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_name(value: Any) -> str:
    value = str(value or "")
    return "development" if value in {"val", "validation", "dev"} else value


def target_is_valid(target: dict[str, Any]) -> tuple[bool, str]:
    direction = str(target.get("direction") or "")
    if direction not in RELATIONS:
        return False, "unsupported_direction"
    mask_path = Path(str(target.get("mask_path") or ""))
    if not mask_path.is_file():
        return False, "missing_dense_label"
    try:
        with Image.open(mask_path) as image:
            mask = image.convert("L")
            actual = bool(mask.getbbox())
    except Exception:
        return False, "dense_label_decode_failure"
    if actual != bool(target.get("has_visual_target")):
        return False, "target_polarity_mismatch"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()
    if args.min_group_size < 2:
        raise SystemExit("--min-group-size must be at least 2")

    source_rows = read_jsonl(args.input)
    if not source_rows:
        raise SystemExit("S2Looking source manifest is empty")
    groups: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    valid_targets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejection_counts: collections.Counter[str] = collections.Counter()
    empty_counts: collections.Counter[str] = collections.Counter()
    pair_ids: set[str] = set()
    split_counts: collections.Counter[str] = collections.Counter()
    mappings: set[tuple[tuple[str, str], ...]] = set()

    for source_row in source_rows:
        pair_id = str(source_row.get("pair_id") or "")
        if not pair_id or pair_id in pair_ids:
            rejection_counts["duplicate_or_missing_pair_id"] += 1
            continue
        pair_ids.add(pair_id)
        if str(source_row.get("dataset_name") or "") != "s2looking":
            rejection_counts["wrong_source_dataset"] += 1
            continue
        split = split_name(source_row.get("split"))
        if split not in {"train", "development", "test"}:
            rejection_counts["unsupported_split"] += 1
            continue
        metadata = source_row.get("source_metadata", {})
        mapping = dict(metadata.get("official_label_mapping", {}))
        mapping["official_source_order"] = metadata.get("official_source_order")
        mappings.add(tuple(sorted((str(k), str(v)) for k, v in mapping.items())))
        if mapping != EXPECTED:
            rejection_counts["official_mapping_mismatch"] += 1
            continue
        by_direction = {str(target.get("direction")): target for target in source_row.get("directional_targets", [])}
        if set(by_direction) != set(RELATIONS):
            rejection_counts["incomplete_directional_targets"] += 1
            continue
        split_counts[split] += 1
        for direction, target in sorted(by_direction.items()):
            valid, reason = target_is_valid(target)
            if not valid:
                rejection_counts[reason] += 1
                continue
            if not bool(target.get("has_visual_target")):
                empty_counts[direction] += 1
                continue
            valid_targets.append((source_row, target))
            groups[(split, direction)].add(pair_id)

    eligible_groups = {key: sorted(values) for key, values in groups.items() if len(values) >= args.min_group_size}
    group_registry = []
    for (split, direction), values in sorted(eligible_groups.items()):
        group_registry.append({
            "schema_version": "qcpr-stage2-structured-semantic-group-v1",
            "semantic_group_id": f"s2looking:official_relation:{direction}",
            "split": split,
            "pair_ids": values,
            "pair_count": len(values),
            "graded_relevance_rule": "own_pair=3;same_official_relation=2",
            "training_enabled": True,
            "verification_status": "structured_source_verified",
            "verification_mode": "official_label_mapping_and_dense_label_contract",
            "human_review_status": "not_applicable_to_official_structured_relation",
        })

    rows_by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    relation_counts: collections.Counter[str] = collections.Counter()
    for source_row, target in valid_targets:
        split = split_name(source_row["split"])
        direction = str(target["direction"])
        positives = eligible_groups.get((split, direction), [])
        if len(positives) < args.min_group_size:
            rejection_counts["relation_group_below_minimum"] += 1
            continue
        pair_id = str(source_row["pair_id"])
        rows_by_split[split].append({
            "schema_version": "qcpr-stage2-structured-semantic-v1",
            "query_id": f"{pair_id}:official_relation:{direction}",
            "canonical_pair_id": pair_id,
            "text": RELATIONS[direction],
            "normalized_text": RELATIONS[direction],
            "split": split,
            "source_dataset": "s2looking",
            "query_scope": "semantic_group",
            "text_granularity": "coarse_relation",
            "semantic_group_id": f"s2looking:official_relation:{direction}",
            "positive_pair_ids": positives if len(positives) <= 256 else [],
            "semantic_group_pair_count": len(positives),
            "self_relevance_grade": 3,
            "other_relevance_grade": 2,
            "graded_relevance_rule": "own_pair=3;same_official_relation=2",
            "ignored_pair_ids": [],
            "training_enabled": True,
            "caption_source": "official_dense_label_semantics",
            "is_generated": True,
            "generator": "official_label_template_v1",
            "verification_status": "structured_source_verified",
            "verification_mode": "official_label_mapping_and_dense_label_contract",
            "human_audit_status": "not_applicable_to_official_structured_relation",
            "dense_label_join_key": pair_id,
            "provenance": {
                "official_source_order": EXPECTED["official_source_order"],
                "official_label": "label1" if direction == "appeared" else "label2",
                "relation": direction,
                "fine_count_location_caption_not_promoted": True,
                "dense_labels_remain_sidecar_only": True,
                "source_pair_id": source_row.get("original_id"),
            },
        })
        relation_counts[direction] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development", "test"):
        output = args.output_dir / f"retrieval_semantic_structured_{split}.jsonl"
        output.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in sorted(rows_by_split.get(split, []), key=lambda row: row["query_id"])), encoding="utf-8")
    (args.output_dir / "semantic_group_registry_structured.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in group_registry), encoding="utf-8")
    audit = {
        "schema_version": "qcpr-stage2-structured-semantic-audit-v1",
        "status": "STRUCTURED_SOURCE_SEMANTIC_READY" if group_registry else "STRUCTURED_SOURCE_SEMANTIC_HOLD",
        "source": "S2Looking",
        "code_sha": args.code_sha,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "official_mapping_required": EXPECTED,
        "observed_mapping_values": [dict(values) for values in sorted(mappings)],
        "source_pair_count": len(source_rows),
        "source_split_counts": dict(sorted(split_counts.items())),
        "eligible_pair_relation_rows": len(valid_targets),
        "output_row_count": sum(len(values) for values in rows_by_split.values()),
        "output_split_counts": {split: len(rows_by_split.get(split, [])) for split in ("train", "development", "test")},
        "output_relation_counts": dict(sorted(relation_counts.items())),
        "group_count": len(group_registry),
        "group_size_distribution": dict(collections.Counter(row["pair_count"] for row in group_registry)),
        "training_enabled": bool(group_registry),
        "human_verified": False,
        "human_review_status": "not_applicable_to_official_structured_relation",
        "excluded_empty_direction_counts": dict(sorted(empty_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "mask_free_manifest": True,
        "mask_paths_in_output": False,
        "dense_label_join": "canonical_pair_id only",
        "notes": [
            "Only coarse relations explicitly encoded by official S2Looking label semantics are promoted.",
            "Generated count/location phrases and RSCC QvQ captions remain outside this view.",
            "This is a separate structured semantic task view, not a human-caption view.",
            "The Stage-2 gate must continue to report generated-caption human review separately.",
        ],
    }
    (args.output_dir / "structured_semantic_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("status", "output_row_count", "output_split_counts", "group_count", "training_enabled", "mask_free_manifest")}, sort_keys=True))
    return 0 if group_registry else 2


if __name__ == "__main__":
    raise SystemExit(main())
