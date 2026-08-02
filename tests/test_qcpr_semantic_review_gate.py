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


def test_human_review_gate_promotes_only_coarse_same_split_groups(tmp_path: Path) -> None:
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
    assert all(row["training_enabled"] is True for row in rows)
    assert all(row["verification_status"] == "human_verified" for row in rows)
    assert all(
        set(row["positive_pair_ids"])
        == {"rscc_ebd:event:item-0", "rscc_ebd:event:item-1"}
        for row in rows
    )
    assert not list(
        (output / "retrieval_semantic_human_verified_development.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    audit = json.loads(
        (output / "semantic_human_review_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "HUMAN_REVIEW_ACCEPTED_MULTI_POSITIVE"
    assert audit["verified_group_count"] == 1
    assert audit["training_enabled"] is True
