#!/usr/bin/env python3
"""Build compact, review-gated graded semantic supervision for RSCC.

Event IDs are used only for split/leakage checks.  Semantic groups are derived
from adjudicated structured attributes and never from event identity.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "development", "test")
REQUIRED_ATTRIBUTES = (
    "disaster_type",
    "changed_object",
    "change_direction",
    "damage_type",
    "severity",
    "spatial_context",
    "count_bucket",
    "verification_confidence",
)
GRADE_3_FIELDS = ("changed_object", "change_direction", "damage_type")
REVIEW_REQUIRED = (
    "visible_change",
    "changed_object",
    "change_direction",
    "damage_type",
    "severity",
    "location_support",
    "count_support",
    "accept_rewrite_reject",
    "confidence",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def normalize(value: Any) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().casefold()
    return value or "unknown"


def empty_outputs(output_dir: Path, review_rows: int, reason: str, code_sha: str | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output_dir / f"retrieval_semantic_gold_{split}.jsonl", [])
    write_jsonl(output_dir / "semantic_group_registry.jsonl", [])
    write_jsonl(output_dir / "verified_rscc_text_registry.jsonl", [])
    write_json(output_dir / "gold_semantic_audit.json", {
        "schema_version": "qcpr-stage2-gold-semantic-audit-v1",
        "status": "GOLD_SEMANTIC_HOLD",
        "code_sha": code_sha,
        "training_enabled": False,
        "review_rows": review_rows,
        "adjudicated_rows": 0,
        "verified_caption_rows": 0,
        "semantic_group_count": 0,
        "reason": reason,
        "event_ids_used_for_semantics": False,
        "positive_grades_enabled": [2, 3],
        "grade_1_weight": 0.1,
    })
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()

    packet = read_jsonl(args.packet)
    reviewer_a = {row["review_id"]: row for row in read_jsonl(args.reviewer_a)}
    reviewer_b = {row["review_id"]: row for row in read_jsonl(args.reviewer_b)}
    adjudicated = {row["review_id"]: row for row in read_jsonl(args.adjudicated)}
    if not packet:
        return empty_outputs(args.output_dir, 0, "human review packet is empty", args.code_sha)
    packet_ids = {row["review_id"] for row in packet}
    if set(reviewer_a) != packet_ids or set(reviewer_b) != packet_ids or set(adjudicated) != packet_ids:
        return empty_outputs(args.output_dir, len(packet), "review decision files do not cover the packet", args.code_sha)
    agreement = read_json(args.agreement)
    if agreement.get("status") != "HUMAN_REVIEW_COMPLETE" or int(agreement.get("adjudicated_rows", 0)) != len(packet):
        return empty_outputs(args.output_dir, len(packet), "two independent human reviews are not complete", args.code_sha)

    records: list[dict[str, Any]] = []
    for row in packet:
        review_id = row["review_id"]
        a, b, final = reviewer_a[review_id], reviewer_b[review_id], adjudicated[review_id]
        for reviewer, decision in (("reviewer_a", a), ("reviewer_b", b)):
            if decision.get("reviewer_id") != reviewer or decision.get("decision_status") != "complete":
                return empty_outputs(args.output_dir, len(packet), f"{reviewer} has pending/incomplete rows", args.code_sha)
            if any(field not in decision for field in REVIEW_REQUIRED):
                return empty_outputs(args.output_dir, len(packet), f"{reviewer} is missing required labels", args.code_sha)
        if final.get("adjudication_status") != "adjudicated" or final.get("final_decision") not in {"accept", "rewrite"}:
            return empty_outputs(args.output_dir, len(packet), "adjudication is incomplete or rejected all rows", args.code_sha)
        attrs = final.get("structured_attributes") or {}
        if any(not normalize(attrs.get(field)) or normalize(attrs.get(field)) == "unknown" for field in REQUIRED_ATTRIBUTES):
            return empty_outputs(args.output_dir, len(packet), "adjudicated structured attributes are incomplete", args.code_sha)
        if normalize(attrs.get("verification_confidence")) not in {"medium", "high"}:
            return empty_outputs(args.output_dir, len(packet), "verified rows require medium/high adjudication confidence", args.code_sha)
        text = str(final.get("final_caption") or row.get("candidate_caption") or "").strip()
        if not text:
            return empty_outputs(args.output_dir, len(packet), "accepted/rewrite row has no final factual caption", args.code_sha)
        if row.get("split") not in SPLITS:
            return empty_outputs(args.output_dir, len(packet), "unsupported split", args.code_sha)
        records.append({
            "review_id": review_id,
            "canonical_pair_id": row["canonical_pair_id"],
            "source_event_id": row["source_event_id"],
            "split": row["split"],
            "t1_path": row["t1_path"],
            "t2_path": row["t2_path"],
            "text": text,
            "attributes": {field: normalize(attrs.get(field)) for field in REQUIRED_ATTRIBUTES},
            "verification_confidence": normalize(attrs["verification_confidence"]),
        })

    # Event IDs are checked for split consistency, but never enter a semantic key.
    events_by_split: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        events_by_split[record["split"]].add(record["source_event_id"])
    if any(len(events_by_split[split] & events_by_split[other]) for split in SPLITS for other in SPLITS if split < other):
        return empty_outputs(args.output_dir, len(packet), "event leakage across splits", args.code_sha)

    groups: dict[tuple[str, str, tuple[str, ...]], list[str]] = collections.defaultdict(list)
    for record in records:
        attrs = record["attributes"]
        exact = tuple(attrs[field] for field in GRADE_3_FIELDS)
        category = (attrs["disaster_type"], attrs["change_direction"], attrs["damage_type"])
        broad = (attrs["disaster_type"], attrs["change_direction"])
        groups[(record["split"], "g3", exact)].append(record["canonical_pair_id"])
        groups[(record["split"], "g2", category)].append(record["canonical_pair_id"])
        groups[(record["split"], "g1", broad)].append(record["canonical_pair_id"])

    registry: list[dict[str, Any]] = []
    group_lookup: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for (split, grade, key), pair_ids in sorted(groups.items()):
        if len(pair_ids) < 2:
            continue
        group_id = f"rscc:semantic:{grade}:{digest('|'.join(key))}"
        group_lookup[(split, grade, key)] = group_id
        registry.append({
            "schema_version": "qcpr-stage2-graded-semantic-group-v1",
            "semantic_group_id": group_id,
            "split": split,
            "grade": int(grade[1]),
            "structured_key": list(key),
            "pair_ids": sorted(set(pair_ids)),
            "pair_count": len(set(pair_ids)),
            "event_ids_are_provenance_only": True,
            "training_enabled": int(grade[1]) in {2, 3},
            "verification_status": "two_reviewer_adjudicated",
        })

    group_by_id = {row["semantic_group_id"]: row for row in registry}
    manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    text_registry: list[dict[str, Any]] = []
    grade_1_audit_only_rows = 0
    for record in records:
        attrs = record["attributes"]
        exact = tuple(attrs[field] for field in GRADE_3_FIELDS)
        category = (attrs["disaster_type"], attrs["change_direction"], attrs["damage_type"])
        broad = (attrs["disaster_type"], attrs["change_direction"])
        refs = []
        for grade, key in (("g3", exact), ("g2", category), ("g1", broad)):
            group_id = group_lookup.get((record["split"], grade, key))
            if group_id:
                refs.append({"semantic_group_id": group_id, "grade": int(grade[1]), "weight": 0.1 if grade == "g1" else 1.0})
        primary = next((ref for ref in refs if ref["grade"] == 3), next((ref for ref in refs if ref["grade"] == 2), None))
        if primary is None:
            if refs:
                grade_1_audit_only_rows += 1
            continue
        row = {
            "schema_version": "temporal-semantic-gold-manifest-v1",
            "query_id": record["review_id"],
            "canonical_pair_id": record["canonical_pair_id"],
            "exact_pair_id": record["canonical_pair_id"],
            "split": record["split"],
            "text": record["text"],
            "semantic_group_id": primary["semantic_group_id"],
            "semantic_group_refs": refs,
            "graded_relevance_policy": "3=same object+direction+damage;2=same change category;1=broad similarity at weight 0.1;0=irrelevant",
            "positive_grades_for_primary_training": [2, 3],
            "same_event_is_not_positive_by_default": True,
            "source_event_id": record["source_event_id"],
            "verification_status": "two_reviewer_adjudicated",
            "verification_confidence": record["verification_confidence"],
            "training_enabled": primary["grade"] in {2, 3},
            "t1_path": record["t1_path"],
            "t2_path": record["t2_path"],
        }
        manifests[record["split"]].append(row)
        text_registry.append({
            "caption_id": record["review_id"],
            "canonical_pair_id": record["canonical_pair_id"],
            "text": record["text"],
            "caption_source": "RSCC-QvQ-human-reviewed",
            "is_generated": True,
            "verification_status": "two_reviewer_adjudicated",
            "verification_confidence": record["verification_confidence"],
            "structured_attributes": attrs,
            "training_enabled": primary["grade"] in {2, 3},
        })

    if not text_registry or not any(row["training_enabled"] for row in text_registry):
        return empty_outputs(args.output_dir, len(packet), "no adjudicated rows formed a training-enabled semantic group", args.code_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.output_dir / f"retrieval_semantic_gold_{split}.jsonl", sorted(manifests[split], key=lambda row: row["query_id"]))
    write_jsonl(args.output_dir / "semantic_group_registry.jsonl", registry)
    write_jsonl(args.output_dir / "verified_rscc_text_registry.jsonl", text_registry)
    write_json(args.output_dir / "gold_semantic_audit.json", {
        "schema_version": "qcpr-stage2-gold-semantic-audit-v1",
        "status": "GOLD_SEMANTIC_READY",
        "code_sha": args.code_sha,
        "training_enabled": True,
        "review_rows": len(packet),
        "adjudicated_rows": len(records),
        "verified_caption_rows": len(text_registry),
        "semantic_group_count": len(registry),
        "group_size_distribution": dict(collections.Counter(row["pair_count"] for row in registry)),
        "manifest_split_counts": {split: len(manifests[split]) for split in SPLITS},
        "event_split_counts": {split: len(events_by_split[split]) for split in SPLITS},
        "event_ids_used_for_semantics": False,
        "positive_grades_enabled": [2, 3],
        "grade_1_weight": 0.1,
        "grade_1_audit_only_rows": grade_1_audit_only_rows,
        "primary_training_grades": [2, 3],
        "same_event_hard_negative_policy": "same-event semantically different pairs remain negatives unless structured grades match",
        "training_enabled_unreviewed_rows": 0,
    })
    print(json.dumps({"status": "GOLD_SEMANTIC_READY", "verified_caption_rows": len(text_registry), "semantic_group_count": len(registry)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
