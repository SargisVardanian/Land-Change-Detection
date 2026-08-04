from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_manual_visual_pilot_builds_true_same_split_multi_positive_rows(tmp_path: Path) -> None:
    source = tmp_path / "review_packet.jsonl"
    rows = [
        {
            "canonical_pair_id": f"rscc_ebd:event:{idx}",
            "split": "train",
            "source_event_id": "event",
            "t1_path": f"/tmp/t1-{idx}.png",
            "t2_path": f"/tmp/t2-{idx}.png",
            "captions": ["unsupported detailed caption"],
            "verification_status": "automated_frozen_siglip2_verified",
        }
        for idx in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "semantic"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_rscc_manual_visual_semantic_pilot.py",
            "--input",
            str(source),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    generated = [
        json.loads(line)
        for line in (output / "retrieval_semantic_manual_visual_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(generated) == 3
    expected = {row["canonical_pair_id"] for row in rows}
    assert all(set(row["positive_pair_ids"]) == expected for row in generated)
    assert all(row["semantic_candidate_count"] == 3 for row in generated)
    report = json.loads(
        (output / "manual_visual_semantic_pilot_audit.json").read_text(encoding="utf-8")
    )
    assert report["multi_positive_rows"] == 3
    assert report["independent_human_audit_rows"] == 0
    assert report["training_enabled"] is False
