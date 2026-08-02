from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_gold_semantic_set_uses_structured_attributes_not_event_ids(tmp_path: Path) -> None:
    rows = []
    for index, (event, obj) in enumerate((("EVENT_A", "building"), ("EVENT_B", "building"), ("EVENT_A", "road"))):
        rows.append({
            "review_id": f"review:{index}", "canonical_pair_id": f"pair:{index}", "source_event_id": event,
            "split": "train", "t1_path": f"/tmp/{index}:pre.png", "t2_path": f"/tmp/{index}:post.png",
            "candidate_caption": f"candidate {index}",
        })
    packet = tmp_path / "packet.jsonl"; _write_jsonl(packet, rows)
    decisions = []
    adjudicated = []
    for row in rows:
        labels = {
            "review_id": row["review_id"], "reviewer_id": "reviewer_a", "decision_status": "complete",
            "visible_change": True, "changed_object": "building" if row["canonical_pair_id"] != "pair:2" else "road",
            "change_direction": "destroyed", "damage_type": "structural", "severity": "high",
            "location_support": False, "count_support": False, "accept_rewrite_reject": "accept", "confidence": "high",
        }
        decisions.append(labels)
        object_name = labels["changed_object"]
        adjudicated.append({
            "review_id": row["review_id"], "adjudication_status": "adjudicated", "final_decision": "accept",
            "final_caption": f"{object_name} was destroyed", "structured_attributes": {
                "disaster_type": "earthquake", "changed_object": object_name, "change_direction": "destroyed",
                "damage_type": "structural", "severity": "high", "spatial_context": "urban",
                "count_bucket": "single", "verification_confidence": "high",
            },
        })
    reviewer_a = tmp_path / "a.jsonl"; reviewer_b = tmp_path / "b.jsonl"; adjud = tmp_path / "adjud.jsonl"
    _write_jsonl(reviewer_a, decisions); _write_jsonl(reviewer_b, [{**row, "reviewer_id": "reviewer_b"} for row in decisions]); _write_jsonl(adjud, adjudicated)
    agreement = tmp_path / "agreement.json"; agreement.write_text(json.dumps({"status": "HUMAN_REVIEW_COMPLETE", "adjudicated_rows": len(rows)}))
    output = tmp_path / "gold"
    result = subprocess.run([
        sys.executable, "scripts/build_qcpr_gold_semantic_set.py", "--packet", str(packet), "--reviewer-a", str(reviewer_a),
        "--reviewer-b", str(reviewer_b), "--adjudicated", str(adjud), "--agreement", str(agreement), "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    assert "GOLD_SEMANTIC_READY" in result.stdout
    audit = json.loads((output / "gold_semantic_audit.json").read_text())
    assert audit["event_ids_used_for_semantics"] is False
    assert audit["training_enabled_unreviewed_rows"] == 0
    groups = [json.loads(line) for line in (output / "semantic_group_registry.jsonl").read_text().splitlines() if line.strip()]
    grade3 = [row for row in groups if row["grade"] == 3]
    assert grade3
    assert set(grade3[0]["pair_ids"]) == {"pair:0", "pair:1"}
    assert "EVENT_A" not in json.dumps(grade3[0])


def test_grade_one_only_review_rows_are_audit_only(tmp_path: Path) -> None:
    rows = []
    for index, (obj, damage) in enumerate((("building", "structural"), ("road", "washout"))):
        rows.append({
            "review_id": f"grade1:{index}",
            "canonical_pair_id": f"grade1-pair:{index}",
            "source_event_id": f"EVENT_{index}",
            "split": "train",
            "t1_path": f"/tmp/{index}:pre.png",
            "t2_path": f"/tmp/{index}:post.png",
            "candidate_caption": f"{obj} changed",
        })
    packet = tmp_path / "packet.jsonl"
    _write_jsonl(packet, rows)

    def reviewer(reviewer_id: str) -> list[dict]:
        return [
            {
                "review_id": row["review_id"],
                "reviewer_id": reviewer_id,
                "decision_status": "complete",
                "visible_change": True,
                "changed_object": obj,
                "change_direction": "damaged",
                "damage_type": damage,
                "severity": "medium",
                "location_support": False,
                "count_support": False,
                "accept_rewrite_reject": "accept",
                "confidence": "high",
            }
            for row, (obj, damage) in zip(rows, (("building", "structural"), ("road", "washout")))
        ]

    adjudicated = [
        {
            "review_id": row["review_id"],
            "adjudication_status": "adjudicated",
            "final_decision": "accept",
            "final_caption": row["candidate_caption"],
            "structured_attributes": {
                "disaster_type": "earthquake",
                "changed_object": obj,
                "change_direction": "damaged",
                "damage_type": damage,
                "severity": "medium",
                "spatial_context": "urban",
                "count_bucket": "unknown",
                "verification_confidence": "high",
            },
        }
        for row, (obj, damage) in zip(rows, (("building", "structural"), ("road", "washout")))
    ]
    reviewer_a = tmp_path / "a.jsonl"
    reviewer_b = tmp_path / "b.jsonl"
    adjud = tmp_path / "adjud.jsonl"
    _write_jsonl(reviewer_a, reviewer("reviewer_a"))
    _write_jsonl(reviewer_b, reviewer("reviewer_b"))
    _write_jsonl(adjud, adjudicated)
    agreement = tmp_path / "agreement.json"
    agreement.write_text(json.dumps({"status": "HUMAN_REVIEW_COMPLETE", "adjudicated_rows": len(rows)}))
    output = tmp_path / "gold"
    subprocess.run([
        sys.executable,
        "scripts/build_qcpr_gold_semantic_set.py",
        "--packet", str(packet),
        "--reviewer-a", str(reviewer_a),
        "--reviewer-b", str(reviewer_b),
        "--adjudicated", str(adjud),
        "--agreement", str(agreement),
        "--output-dir", str(output),
    ], check=False)
    audit = json.loads((output / "gold_semantic_audit.json").read_text())
    assert audit["status"] == "GOLD_SEMANTIC_HOLD"
    assert audit["training_enabled"] is False
    assert audit["positive_grades_enabled"] == [2, 3]
    assert all(
        not (output / f"retrieval_semantic_gold_{split}.jsonl").read_text().strip()
        for split in ("train", "development", "test")
    )
