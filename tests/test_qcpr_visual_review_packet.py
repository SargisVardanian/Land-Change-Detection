from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_visual_packet_is_human_review_pending_and_not_semantic_supervision(tmp_path: Path) -> None:
    rows = []
    for event_index, event in enumerate(("EVENT_A", "EVENT_B")):
        pre = tmp_path / f"{event}_pre.png"
        post = tmp_path / f"{event}_post.png"
        Image.new("RGB", (12, 12), (event_index * 40, 20, 80)).save(pre)
        Image.new("RGB", (12, 12), (event_index * 40, 80, 20)).save(post)
        for pair_index in range(2):
            rows.append({
                "canonical_pair_id": f"rscc_ebd:{event}:{pair_index}",
                "source_event_id": event,
                "t1_path": str(pre),
                "t2_path": str(post),
                "captions": [f"candidate {event} {pair_index}"],
                "verification_score": 0.5,
            })
    source = tmp_path / "candidates.jsonl"
    _write_jsonl(source, rows)
    output = tmp_path / "visual"
    subprocess.run([
        sys.executable,
        "scripts/build_rscc_visual_review_packet.py",
        "--input", str(source),
        "--output-dir", str(output),
        "--per-event", "1",
    ], check=True, capture_output=True, text=True)
    audit = json.loads((output / "review_packet_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "qcpr-stage2-rscc-visual-review-packet-v2"
    assert audit["rows"] == 2
    assert audit["reviewer_status"] == "pending_independent_human_review"
    assert audit["human_audit_passed"] is False
    assert audit["event_ids_provenance_only"] is True
    assert audit["semantic_positive_sets_materialized"] is False
    assert len(audit["sheets"]) == 1
    assert (output / "README.md").is_file()
