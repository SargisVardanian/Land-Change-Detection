#!/usr/bin/env python3
"""Build an automated-verifier semantic candidate view with compact groups."""
from __future__ import annotations
import argparse
import collections
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
    groups = collections.defaultdict(list)
    for row in accepted:
        key = (
            str(row.get("semantic_group_id") or f"rscc_ebd:event:{row.get('source_event_id') or 'unknown'}"),
            str(row.get("split") or "unknown"),
        )
        groups[key].append(str(row["canonical_pair_id"]))
    groups = {key: sorted(set(values)) for key, values in groups.items()}

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, src in enumerate(accepted):
        group_key = (
            str(src.get("semantic_group_id") or f"rscc_ebd:event:{src.get('source_event_id') or 'unknown'}"),
            str(src.get("split") or "unknown"),
        )
        group_pairs = groups[group_key]
        if len(group_pairs) < 2:
            continue
        event = str(src.get("source_event_id") or "unknown")
        pair = str(src["canonical_pair_id"])
        text = str((src.get("captions") or [""])[0]).strip()
        rows.append(
            {
                "schema_version": "qcpr-stage2-semantic-candidate-v2",
                "query_id": f"{pair}:automated_verified:{index:05d}",
                "canonical_pair_id": pair,
                "text": text,
                "normalized_text": " ".join(text.casefold().split()),
                "split": str(src["split"]),
                "source_dataset": "RSCC-EBD",
                "query_scope": "semantic_group",
                "semantic_group_id": group_key[0],
                "positive_pair_ids": group_pairs,
                "semantic_candidate_count": len(group_pairs),
                "self_relevance_grade": 3,
                "other_relevance_grade": 1,
                "graded_relevance_rule": "own_pair=3;same_event_broad_relation=1",
                "caption_source": "RSCC-QvQ",
                "is_generated": True,
                "verification_status": "automated_frozen_siglip2_verified",
                "verification_method": "SigLIP2 image-text consistency with human LEVIR/SECOND calibration",
                "verification_score": src.get("verification_score"),
                "verification_shuffled_score": src.get("verification_shuffled_score"),
                "training_enabled": False,
                "human_audit_status": "required",
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

    by_split = collections.Counter(row["split"] for row in rows)
    group_registry = [
        {
            "semantic_group_id": key[0],
            "split": key[1],
            "pair_ids": values,
            "pair_count": len(values),
            "graded_relevance_rule": "own_pair=3;same_event_broad_relation=1",
            "training_enabled": False,
        }
        for key, values in sorted(groups.items())
        if len(values) >= 2
    ]
    for split in sorted(by_split):
        path = out / f"retrieval_semantic_automated_verified_{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows if row["split"] == split),
            encoding="utf-8",
        )
    (out / "semantic_group_registry.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in group_registry),
        encoding="utf-8",
    )
    by_group = collections.Counter(row["semantic_group_id"] for row in rows)
    report = {
        "schema_version": "qcpr-stage2-automated-verified-semantic-pilot-v2",
        "status": "AUTOMATED_VERIFIED_HUMAN_REVIEW_REQUIRED",
        "row_count": len(rows),
        "split_counts": dict(sorted(by_split.items())),
        "semantic_group_count": len(group_registry),
        "group_size_distribution": dict(collections.Counter(row["pair_count"] for row in group_registry)),
        "multi_positive_rows": len(rows),
        "single_positive_rows": 0,
        "training_enabled": False,
        "human_audit_passed": False,
        "source_verification": "frozen SigLIP2 consistency screen",
        "notes": [
            "Each row has all same-split pairs in its event group as candidate positives.",
            "The verifier does not establish exact equivalence.",
            "Human stratified audit is still required before training.",
        ],
    }
    (out / "automated_verified_semantic_pilot_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"row_count": len(rows), "split_counts": dict(sorted(by_split.items())), "semantic_group_count": len(group_registry), "status": report["status"]}, sort_keys=True))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
