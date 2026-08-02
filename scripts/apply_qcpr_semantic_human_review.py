#!/usr/bin/env python3
"""Promote only independently human-reviewed semantic coarse decisions.

The script refuses partial, duplicated, automated, or split-inconsistent
decisions. It never promotes detailed generated captions as exact relevance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_COARSE = {"accept", "reject", "uncertain"}
ALLOWED_FINE = {"accept", "reject", "not_reviewed", "uncertain", ""}
REJECT_REVIEWER_TYPES = {"automated", "codex", "model", "qvq", "siglip2"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    packet = read_jsonl(args.packet)
    decisions = read_jsonl(args.decisions)
    packet_ids = [str(row["canonical_pair_id"]) for row in packet]
    if not packet_ids or len(packet_ids) != len(set(packet_ids)):
        raise SystemExit("packet must contain unique canonical_pair_id values")
    decision_ids = [str(row.get("canonical_pair_id") or "") for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise SystemExit("decisions contain duplicate canonical_pair_id values")
    if set(packet_ids) != set(decision_ids):
        missing = sorted(set(packet_ids) - set(decision_ids))
        extra = sorted(set(decision_ids) - set(packet_ids))
        raise SystemExit(f"decision coverage mismatch: missing={missing[:5]} extra={extra[:5]}")

    by_id = {str(row["canonical_pair_id"]): row for row in decisions}
    accepted: list[dict[str, Any]] = []
    counts = collections.Counter()
    reviewer_ids: set[str] = set()
    for packet_row in packet:
        pair_id = str(packet_row["canonical_pair_id"])
        decision = by_id[pair_id]
        reviewer_id = str(decision.get("reviewer_id") or "").strip()
        reviewer_type = str(decision.get("reviewer_type") or "").casefold().strip()
        coarse = str(decision.get("coarse_change_decision") or "").casefold().strip()
        fine = str(decision.get("fine_caption_decision") or "").casefold().strip()
        contradiction = bool(decision.get("contradiction_flag", False))
        if not reviewer_id or reviewer_type != "human" or reviewer_type in REJECT_REVIEWER_TYPES:
            raise SystemExit(f"non-human or missing reviewer for {pair_id}")
        if coarse not in ALLOWED_COARSE:
            raise SystemExit(f"invalid coarse_change_decision for {pair_id}: {coarse!r}")
        if fine not in ALLOWED_FINE:
            raise SystemExit(f"invalid fine_caption_decision for {pair_id}: {fine!r}")
        if not str(decision.get("reviewed_at") or "").strip():
            raise SystemExit(f"reviewed_at is required for {pair_id}")
        reviewer_ids.add(reviewer_id)
        counts[coarse] += 1
        if coarse == "accept" and not contradiction and fine in {"reject", "not_reviewed", ""}:
            accepted.append(
                {
                    "packet": packet_row,
                    "decision": decision,
                    "reviewer_id": reviewer_id,
                }
            )
        elif coarse == "accept":
            counts["accept_rejected_by_contradiction_or_fine"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in accepted:
        by_split[str(row["packet"].get("split") or "unknown")].append(row)

    verified_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    group_registry: list[dict[str, Any]] = []
    for split, members in sorted(by_split.items()):
        pair_ids = sorted({str(item["packet"]["canonical_pair_id"]) for item in members})
        if len(pair_ids) < 2:
            continue
        group_id = f"rscc_ebd:human_verified:coarse_disaster_change:{split}"
        group_registry.append(
            {
                "semantic_group_id": group_id,
                "split": split,
                "pair_ids": pair_ids,
                "pair_count": len(pair_ids),
                "graded_relevance_rule": "own_pair=3;same_coarse_change=1",
                "training_enabled": True,
                "verification_status": "human_verified",
                "reviewer_ids": sorted({item["reviewer_id"] for item in members}),
            }
        )
        for item in members:
            packet_row = item["packet"]
            pair_id = str(packet_row["canonical_pair_id"])
            verified_rows[split].append(
                {
                    "schema_version": "qcpr-stage2-semantic-human-verified-v1",
                    "query_id": f"{pair_id}:human_verified_coarse",
                    "canonical_pair_id": pair_id,
                    "text": "a visible disaster-related change occurs between the two dates",
                    "normalized_text": "a visible disaster-related change occurs between the two dates",
                    "split": split,
                    "source_dataset": "RSCC-EBD",
                    "query_scope": "semantic_group",
                    "text_granularity": "coarse_relation",
                    "semantic_group_id": group_id,
                    "positive_pair_ids": pair_ids if len(pair_ids) <= 256 else [],
                    "semantic_group_pair_count": len(pair_ids),
                    "self_relevance_grade": 3,
                    "other_relevance_grade": 1,
                    "graded_relevance_rule": "own_pair=3;same_coarse_change=1",
                    "ignored_pair_ids": [],
                    "training_enabled": True,
                    "verification_status": "human_verified",
                    "human_audit_status": "passed",
                    "reviewer_ids": sorted({item["reviewer_id"] for item in members}),
                    "provenance": {
                        "packet_pair_id": pair_id,
                        "original_generated_caption": (packet_row.get("captions") or [""])[0],
                        "fine_caption_promoted": False,
                        "decision_file_sha256": sha256(args.decisions),
                        "review_basis": "independent human T1/T2 coarse temporal-change decision",
                    },
                }
            )

    for split in ("train", "development", "test"):
        path = args.output_dir / f"retrieval_semantic_human_verified_{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in sorted(verified_rows.get(split, []), key=lambda row: row["query_id"])
            ),
            encoding="utf-8",
        )
    (args.output_dir / "semantic_group_registry_human_verified.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in group_registry),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "qcpr-stage2-semantic-human-review-audit-v1",
        "status": (
            "HUMAN_REVIEW_ACCEPTED_MULTI_POSITIVE"
            if group_registry
            else "HUMAN_REVIEW_COMPLETE_NO_MULTI_POSITIVE_GROUP"
        ),
        "packet_rows": len(packet),
        "decision_rows": len(decisions),
        "accepted_coarse_rows": len(accepted),
        "verified_rows": sum(len(values) for values in verified_rows.values()),
        "verified_group_count": len(group_registry),
        "split_counts": {split: len(values) for split, values in sorted(verified_rows.items())},
        "decision_counts": dict(sorted(counts.items())),
        "reviewer_ids": sorted(reviewer_ids),
        "training_enabled": bool(group_registry),
        "packet_sha256": sha256(args.packet),
        "decisions_sha256": sha256(args.decisions),
        "notes": [
            "Only coarse change decisions are promoted.",
            "Fine generated QvQ detail is never promoted by this gate.",
            "All positive pairs remain inside the official split.",
            "This output is eligible for semantic training only when training_enabled is true and the parent release records the audit hash.",
        ],
    }
    (args.output_dir / "semantic_human_review_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
