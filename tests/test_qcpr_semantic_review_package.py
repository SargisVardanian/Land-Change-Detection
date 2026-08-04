from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_review_package_is_deterministic_and_not_training_enabled(tmp_path: Path) -> None:
    pairs = []
    candidates = []
    for event_index, event in enumerate(("EVENT_A", "EVENT_B")):
        split = "train" if event_index == 0 else "development"
        for index in range(3):
            pair_id = f"rscc_ebd:{event}:{index}"
            pairs.append({
                "canonical_pair_id": pair_id,
                "source_event_id": event,
                "source_scene_group_id": f"rscc_ebd:event:{event}",
                "split": split,
                "t1_path": f"/tmp/{pair_id}:pre.png",
                "t2_path": f"/tmp/{pair_id}:post.png",
            })
            candidates.append({"canonical_pair_id": pair_id, "captions": [f"candidate {event} {index}"], "caption_source": "RSCC-QvQ", "generator": "QvQ-Max"})
    pairs_path = tmp_path / "pairs.jsonl"; _write_jsonl(pairs_path, pairs)
    candidates_path = tmp_path / "candidates.jsonl"; _write_jsonl(candidates_path, candidates)
    output = tmp_path / "review"
    command = [sys.executable, "scripts/build_qcpr_semantic_review_package.py", "--pairs", str(pairs_path), "--candidates", str(candidates_path), "--output-dir", str(output), "--seed", "7", "--per-event", "2", "--expected-events", "2"]
    subprocess.run(command, check=True)
    first = (output / "human_review_packet.jsonl").read_bytes()
    second = tmp_path / "review_second"
    command_second = list(command)
    command_second[command_second.index(str(output))] = str(second)
    subprocess.run(command_second, check=True)
    assert first == (second / "human_review_packet.jsonl").read_bytes()
    packet = [json.loads(line) for line in first.decode().splitlines() if line.strip()]
    assert len(packet) == 4
    assert {row["source_event_id"] for row in packet} == {"EVENT_A", "EVENT_B"}
    agreement = json.loads((output / "human_review_agreement.json").read_text())
    assert agreement["status"] == "PENDING_HUMAN_REVIEW"
    assert agreement["training_enabled_rows"] == 0
    assert all(json.loads(line)["decision_status"] == "pending" for line in (output / "reviewer_a_decisions.jsonl").read_text().splitlines())
