from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_single_reviewer_coarse_review_never_promotes_event_groups(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    output = tmp_path / "verified"
    packet_rows = [
        {
            "canonical_pair_id": f"rscc_ebd:event:item-{index}",
            "split": split,
            "source_event_id": "event-a",
            "captions": [f"detailed generated caption {index}"],
        }
        for index, split in enumerate(("train", "train", "development"))
    ]
    decision_rows = [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            "reviewer_id": "human-reviewer-1",
            "reviewer_type": "human",
            "coarse_change_decision": "accept",
            "fine_caption_decision": "reject",
            "contradiction_flag": False,
            "reviewed_at": "2026-08-02T12:00:00Z",
            "notes": "coarse change visible",
        }
        for row in packet_rows
    ]
    _write_jsonl(packet, packet_rows)
    _write_jsonl(decisions, decision_rows)
    subprocess.run(
        [
            sys.executable,
            "scripts/apply_qcpr_semantic_human_review.py",
            "--packet",
            str(packet),
            "--decisions",
            str(decisions),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (output / "retrieval_semantic_human_verified_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(row["training_enabled"] is False for row in rows)
    assert all(row["verification_status"] == "human_coarse_review_only" for row in rows)
    assert all(row["query_scope"] == "semantic_audit_candidate" for row in rows)
    assert all(row["semantic_group_id"] is None for row in rows)
    assert all(row["positive_pair_ids"] == [] for row in rows)
    assert all(row["event_ids_are_provenance_only"] is True for row in rows)
    assert not list(
        (output / "semantic_group_registry_human_verified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    development_audit_rows = [
        json.loads(line)
        for line in (output / "retrieval_semantic_human_verified_development.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(development_audit_rows) == 1
    assert development_audit_rows[0]["training_enabled"] is False
    audit = json.loads(
        (output / "semantic_human_review_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "HUMAN_REVIEW_COARSE_ONLY_HOLD"
    assert audit["verified_group_count"] == 0
    assert audit["coarse_audit_only_rows"] == 3
    assert audit["event_ids_used_for_semantics"] is False
    assert audit["training_enabled"] is False
