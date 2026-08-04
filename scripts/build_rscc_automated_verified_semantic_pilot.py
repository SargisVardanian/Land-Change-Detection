#!/usr/bin/env python3
"""Record automated RSCC checks without promoting semantic positives.

An automated SigLIP2 consistency score is not an independent semantic review.
In particular, an event ID cannot define a positive set.  This script keeps a
small audit-only record and emits no semantic training rows or group registry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    accepted = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted.sort(key=lambda row: (str(row.get("split")), str(row.get("canonical_pair_id"))))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, src in enumerate(accepted):
        event = str(src.get("source_event_id") or "unknown")
        pair = str(src["canonical_pair_id"])
        text = str((src.get("captions") or [""])[0]).strip()
        rows.append(
            {
                "schema_version": "qcpr-stage2-semantic-audit-only-v3",
                "query_id": f"{pair}:automated_verified:{index:05d}",
                "canonical_pair_id": pair,
                "text": text,
                "normalized_text": " ".join(text.casefold().split()),
                "split": str(src["split"]),
                "source_dataset": "RSCC-EBD",
                "query_scope": "semantic_audit_candidate",
                "semantic_group_id": None,
                "positive_pair_ids": [],
                "semantic_candidate_count": 0,
                "self_relevance_grade": None,
                "other_relevance_grade": None,
                "graded_relevance_rule": None,
                "caption_source": "RSCC-QvQ",
                "is_generated": True,
                "verification_status": "automated_frozen_siglip2_verified",
                "verification_method": "SigLIP2 image-text consistency with human LEVIR/SECOND calibration",
                "verification_score": src.get("verification_score"),
                "verification_shuffled_score": src.get("verification_shuffled_score"),
                "training_enabled": False,
                "human_audit_status": "required_two_reviewer_structured_review",
                "event_ids_are_provenance_only": True,
                "t1_path": src.get("t1_path"),
                "t2_path": src.get("t2_path"),
                "provenance": {
                    "source_event_id": event,
                    "model": "siglip2-base-patch16-256",
                    "source_row_status": src.get("verification_status"),
                    "independent_human_audit_passed": False,
                },
            }
        )

    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], 0)
        by_split[row["split"]] += 1
    group_registry = []
    for split in ("train", "development", "test"):
        path = out / f"retrieval_semantic_automated_verified_{split}.jsonl"
        path.write_text(
            "",
            encoding="utf-8",
        )
    (out / "semantic_group_registry.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in group_registry),
        encoding="utf-8",
    )
    report = {
        "schema_version": "qcpr-stage2-automated-verified-semantic-audit-v3",
        "status": "AUTOMATED_VERIFIED_AUDIT_ONLY_HOLD",
        "audit_candidate_count": len(rows),
        "row_count": 0,
        "split_counts": dict(sorted(by_split.items())),
        "semantic_group_count": 0,
        "group_size_distribution": {},
        "multi_positive_rows": 0,
        "single_positive_rows": 0,
        "training_enabled": False,
        "human_audit_passed": False,
        "source_verification": "frozen SigLIP2 consistency screen",
        "event_ids_used_for_semantics": False,
        "event_only_groups": 0,
        "notes": [
            "SigLIP2-only verification does not establish structured semantic equivalence.",
            "Event IDs are provenance only and do not define positive sets.",
            "All candidates require two independent human reviews and structured adjudication before semantic training.",
        ],
    }
    (out / "automated_verified_semantic_pilot_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"row_count": 0, "audit_candidate_count": len(rows), "split_counts": dict(sorted(by_split.items())), "semantic_group_count": 0, "status": report["status"]}, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
