#!/usr/bin/env python3
"""Build a deterministic two-reviewer RSCC semantic review package.

The package is deliberately decision-free.  It samples 20 pairs per event,
creates two independent pending decision sheets, and never promotes an
automated or Codex inspection to human verification.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_LABELS = (
    "disaster_type",
    "visible_change",
    "changed_object",
    "change_direction",
    "damage_type",
    "severity",
    "spatial_context",
    "location_support",
    "count_bucket",
    "count_support",
    "accept_rewrite_reject",
    "confidence",
)
VALID_DECISIONS = ("accept", "rewrite", "reject")
VALID_CONFIDENCE = ("low", "medium", "high")
SPLITS = ("train", "development", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def stable_rank(seed: int, event_id: str, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{event_id}:{pair_id}".encode("utf-8")).hexdigest()


def pending_labels(reviewer_id: str) -> dict[str, Any]:
    return {
        "reviewer_id": reviewer_id,
        # A role name is not evidence that two people reviewed independently.
        # Reviewers fill an opaque, stable human identity in the agreement file
        # and repeat it in each completed row.
        "reviewer_identity": None,
        "decision_status": "pending",
        "reviewed_at": None,
        "independence_attestation": None,
        "visible_change": None,
        "changed_object": None,
        "change_direction": None,
        "damage_type": None,
        "severity": None,
        "spatial_context": None,
        "location_support": None,
        "count_bucket": None,
        "count_support": None,
        "accept_rewrite_reject": None,
        "rewritten_caption": None,
        "confidence": None,
        "notes": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--per-event", type=int, default=20)
    parser.add_argument("--expected-events", type=int, default=12)
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()
    if args.per_event < 1:
        raise SystemExit("--per-event must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty review package: {args.output_dir}")

    pair_rows = read_jsonl(args.pairs)
    candidate_rows = {str(row["canonical_pair_id"]): row for row in read_jsonl(args.candidates)}
    by_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen_pairs: set[str] = set()
    for row in pair_rows:
        pair_id = str(row.get("canonical_pair_id") or "")
        event_id = str(row.get("source_event_id") or "")
        if not pair_id or not event_id or pair_id in seen_pairs:
            raise SystemExit(f"invalid or duplicate pair row: {pair_id}")
        if str(row.get("split")) not in SPLITS:
            raise SystemExit(f"unsupported split for {pair_id}: {row.get('split')}")
        if pair_id not in candidate_rows:
            raise SystemExit(f"missing QvQ candidate for {pair_id}")
        seen_pairs.add(pair_id)
        by_event[event_id].append(row)
    if args.expected_events > 0 and len(by_event) != args.expected_events:
        raise SystemExit(f"expected {args.expected_events} RSCC events, observed {len(by_event)}")
    if any(len(rows) < args.per_event for rows in by_event.values()):
        raise SystemExit("at least one event has fewer than the requested review sample")

    packet: list[dict[str, Any]] = []
    for event_id in sorted(by_event):
        selected = sorted(
            by_event[event_id],
            key=lambda row: stable_rank(args.seed, event_id, str(row["canonical_pair_id"])),
        )[: args.per_event]
        for row in selected:
            pair_id = str(row["canonical_pair_id"])
            candidate = candidate_rows[pair_id]
            caption = str((candidate.get("captions") or [""])[0])
            review_id = f"rscc_review_v2:{event_id}:{pair_id.split(':')[-1]}"
            packet.append({
                "schema_version": "qcpr-stage2-human-review-row-v2",
                "review_id": review_id,
                "canonical_pair_id": pair_id,
                "source_event_id": event_id,
                "source_scene_group_id": row.get("source_scene_group_id"),
                "split": row["split"],
                "t1_path": row["t1_path"],
                "t2_path": row["t2_path"],
                "candidate_caption": caption,
                "caption_source": candidate.get("caption_source"),
                "generator": candidate.get("generator"),
                "provenance_only_event": event_id,
                "event_must_not_define_semantic_positive": True,
                "required_labels": list(REQUIRED_LABELS),
                "allowed_decisions": list(VALID_DECISIONS),
                "allowed_confidence": list(VALID_CONFIDENCE),
                "review_status": "pending_two_independent_reviewers",
            })
    packet.sort(key=lambda row: row["review_id"])
    if len(packet) != args.per_event * len(by_event):
        raise SystemExit("review packet cardinality mismatch")

    reviewer_a = [{**row, **pending_labels("reviewer_a")} for row in packet]
    reviewer_b = [{**row, **pending_labels("reviewer_b")} for row in packet]
    adjudicated = [{
        "schema_version": "qcpr-stage2-adjudicated-review-row-v2",
        "review_id": row["review_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "source_event_id": row["source_event_id"],
        "split": row["split"],
        "adjudication_status": "pending_independent_reviews",
        "reviewer_a_decision_status": "pending",
        "reviewer_b_decision_status": "pending",
        "final_decision": None,
        "structured_attributes": None,
        "verification_confidence": None,
    } for row in packet]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "human_review_packet.jsonl", packet)
    write_jsonl(args.output_dir / "reviewer_a_decisions.jsonl", reviewer_a)
    write_jsonl(args.output_dir / "reviewer_b_decisions.jsonl", reviewer_b)
    write_jsonl(args.output_dir / "adjudicated_decisions.jsonl", adjudicated)
    write_jsonl(args.output_dir / "verified_rscc_text_registry.jsonl", [])
    write_json(args.output_dir / "human_review_agreement.json", {
        "schema_version": "qcpr-stage2-human-review-agreement-v2",
        "status": "PENDING_HUMAN_REVIEW",
        "reviewer_count": 2,
        "reviewers": ["reviewer_a", "reviewer_b"],
        "reviewer_identities": {"reviewer_a": None, "reviewer_b": None},
        "independent_review_attested": False,
        "adjudicator_identity": None,
        "rows_sampled": len(packet),
        "events_sampled": len(by_event),
        "rows_reviewed": 0,
        "agreement": None,
        "cohen_kappa": None,
        "adjudicated_rows": 0,
        "training_enabled_rows": 0,
        "codex_visual_inspection_counted_as_human_review": False,
    })
    write_json(args.output_dir / "review_package_audit.json", {
        "schema_version": "qcpr-stage2-human-review-package-audit-v2",
        "status": "READY_FOR_TWO_INDEPENDENT_HUMAN_REVIEWERS",
        "code_sha": args.code_sha,
        "seed": args.seed,
        "per_event": args.per_event,
        "event_count": len(by_event),
        "row_count": len(packet),
        "event_counts": {event: args.per_event for event in sorted(by_event)},
        "split_counts": dict(collections.Counter(row["split"] for row in packet)),
        "pair_ids_unique": len({row["canonical_pair_id"] for row in packet}) == len(packet),
        "event_ids_provenance_only": True,
        "semantic_positive_sets_materialized": False,
        "training_enabled": False,
        "reviewer_decisions_required": ["reviewer_a", "reviewer_b"],
        "reviewer_identity_fields_required": [
            "reviewer_identity",
            "reviewed_at",
            "independence_attestation",
        ],
        "required_labels": list(REQUIRED_LABELS),
        "source_pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "candidate_captions_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
    })
    (args.output_dir / "README.md").write_text(
        "# RSCC Stage-2 human review package\n\n"
        "Reviewers must inspect both T1 and T2. Event IDs are provenance only; "
        "they must never define semantic positives. Complete reviewer_a_decisions.jsonl "
        "and reviewer_b_decisions.jsonl independently. Codex or automated model "
        "inspection is not a human review.\n\n"
        "Before submission, fill human_review_agreement.json with two distinct "
        "opaque human reviewer identities and set independent_review_attested=true. "
        "Each completed decision row must repeat its reviewer_identity, include an "
        "ISO-8601 reviewed_at timestamp, and set independence_attestation=true. "
        "Do not use the same person for both identities. Adjudication and training "
        "promotion remain disabled until both complete sheets are returned and the "
        "gold builder validates this provenance.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "READY_FOR_TWO_INDEPENDENT_HUMAN_REVIEWERS", "events": len(by_event), "rows": len(packet), "training_enabled": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
