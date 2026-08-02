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


def test_stage2_builder_emits_same_split_multi_positive_candidates(tmp_path: Path) -> None:
    pairs = [
        {
            "canonical_pair_id": f"rscc_ebd:event:pair-{index}",
            "split": split,
            "t1_path": f"/tmp/{index}-t1.png",
            "t2_path": f"/tmp/{index}-t2.png",
        }
        for index, split in enumerate(("train", "train", "development"))
    ]
    qvq = [
        {
            "canonical_pair_id": pair["canonical_pair_id"],
            "split": pair["split"],
            "source_event_id": "event-a",
            "captions": [f"detailed caption {index}"],
        }
        for index, pair in enumerate(pairs)
    ]
    pair_registry = tmp_path / "pairs.jsonl"
    caption_registry = tmp_path / "captions.jsonl"
    qvq_path = tmp_path / "qvq.jsonl"
    _write_jsonl(pair_registry, pairs)
    _write_jsonl(caption_registry, [])
    _write_jsonl(qvq_path, qvq)
    output = tmp_path / "semantic"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_qcpr_stage2_semantic_manifests.py",
            "--caption-registry",
            str(caption_registry),
            "--pair-registry",
            str(pair_registry),
            "--rscc-qvq",
            str(qvq_path),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (output / "retrieval_semantic_train_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected = {
        "rscc_ebd:event:pair-0",
        "rscc_ebd:event:pair-1",
    }
    assert len(rows) == 2
    assert all(set(row["positive_pair_ids"]) == expected for row in rows)
    assert all(row["semantic_candidate_count"] == 2 for row in rows)
    assert all(row["self_relevance_grade"] == 3 for row in rows)
    assert all(row["other_relevance_grade"] == 1 for row in rows)
    group_rows = [
        json.loads(line)
        for line in (output / "semantic_group_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert group_rows == [
        {
            "graded_relevance_rule": "pair=3;same_event=1",
            "pair_count": 2,
            "pair_ids": sorted(expected),
            "semantic_group_id": "rscc_ebd:event:event-a",
            "split": "train",
            "training_enabled": False,
        }
    ]
    assert all(row["training_enabled"] is False for row in rows)
    assert all(str(row["human_audit_status"]).startswith("required") for row in rows)
