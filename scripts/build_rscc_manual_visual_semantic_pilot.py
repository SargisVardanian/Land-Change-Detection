#!/usr/bin/env python3
"""Create a conservative coarse semantic pilot from the visual review packet."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path

VERIFIED_TEXT = "A visible disaster-related change occurs between the two dates."
GROUP = "rscc_ebd:manual_visual:disaster_related_change"

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    src = read_jsonl(args.input)
    if not src:
        raise SystemExit("review packet is empty")
    pair_ids = [str(item["canonical_pair_id"]) for item in src]
    if len(set(pair_ids)) != len(pair_ids):
        raise SystemExit("review packet contains duplicate canonical_pair_id values")
    pairs_by_split = {
        split: sorted(
            {
                str(item["canonical_pair_id"])
                for item in src
                if str(item["split"]) == split
            }
        )
        for split in ("train", "development", "test")
    }
    rows, decisions = [], []
    for item in src:
        pair = str(item["canonical_pair_id"])
        split = str(item["split"])
        positives = pairs_by_split.get(split, [])
        if len(positives) < 2:
            raise SystemExit(f"coarse semantic group is not multi-positive in split {split!r}")
        row = {
            "schema_version": "qcpr-stage2-semantic-manual-visual-pilot-v2",
            "query_id": f"{pair}:manual_visual_coarse",
            "canonical_pair_id": pair,
            "text": VERIFIED_TEXT,
            "normalized_text": " ".join(VERIFIED_TEXT.casefold().split()),
            "split": split,
            "source_dataset": "RSCC-EBD",
            "query_scope": "semantic_group",
            "semantic_group_id": GROUP,
            "positive_pair_ids": positives,
            "graded_relevance_rule": "same_coarse_disaster_change=1; exact_pair=0",
            "semantic_candidate_count": len(positives),
            "caption_source": "manual_visual_audit",
            "is_generated": True,
            "generator": "codex_visual_audit",
            "verification_status": "manual_visual_coarse_verified",
            "verification_method": "manual T1/T2 visual comparison; coarse claim only",
            "verification_score": None,
            "verification_shuffled_score": None,
            "training_enabled": False,
            "human_audit_status": "not_human_reviewed",
            "fine_caption_status": "rejected_unverified_detail",
            "t1_path": item.get("t1_path"),
            "t2_path": item.get("t2_path"),
            "provenance": {
                "source_event_id": item.get("source_event_id"),
                "source_pair_id": item.get("source_pair_id"),
                "original_generated_caption": (item.get("captions") or [None])[0],
                "original_verification_status": item.get("verification_status"),
                "review_decision": "accept_coarse_reject_fine",
                "positive_construction": "all distinct coarse-reviewed rows in the same official split",
            },
        }
        rows.append(row)
        decisions.append({
            "canonical_pair_id": pair,
            "split": split,
            "decision": "ACCEPT_COARSE_REJECT_FINE",
            "coarse_claim": VERIFIED_TEXT,
            "fine_caption_decision": "REJECT_UNVERIFIED_DETAIL",
            "reviewer": "Codex visual audit",
            "review_basis": "visible temporal difference in T1/T2; no endorsement of object/count/location claims",
            "original_caption": (item.get("captions") or [None])[0],
            "verification_score": item.get("verification_score"),
            "positive_pair_count": len(positives),
        })
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development", "test"):
        vals = [r for r in rows if r["split"] == split]
        (out / f"retrieval_semantic_manual_visual_{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in vals),
            encoding="utf-8",
        )
    (out / "manual_visual_decisions.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in decisions),
        encoding="utf-8",
    )
    positive_counts = {
        split: len(pairs)
        for split, pairs in pairs_by_split.items()
        if pairs
    }
    report = {
        "schema_version": "qcpr-stage2-semantic-manual-visual-pilot-audit-v2",
        "status": "MANUAL_VISUAL_COARSE_PILOT_HUMAN_REVIEW_STILL_REQUIRED",
        "rows": len(rows),
        "split_counts": dict(sorted(collections.Counter(r["split"] for r in rows).items())),
        "semantic_group_count": 1,
        "semantic_group_id": GROUP,
        "verified_semantic_rows": len(rows),
        "multi_positive_rows": len(rows),
        "positive_pair_counts_by_split": positive_counts,
        "manual_visual_audit_rows": len(rows),
        "independent_human_audit_rows": 0,
        "fine_generated_captions_accepted": 0,
        "fine_generated_captions_rejected": len(rows),
        "training_enabled": False,
        "source_packet_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "notes": [
            "This is a conservative visual pilot, not a human audit.",
            "Only a broad visible temporal-change claim is accepted.",
            "Each row uses the same-split coarse-positive set; it is not exact-pair supervision.",
            "Original QvV detail is preserved for provenance but is excluded from supervision.",
            "Do not promote to final semantic training until an independent human audit is recorded.",
        ],
    }
    (out / "manual_visual_semantic_pilot_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
