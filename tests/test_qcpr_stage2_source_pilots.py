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


def test_forest_scene_proposal_is_disjoint_and_unreleased(tmp_path: Path) -> None:
    rows = []
    for index, group in enumerate(("scene-a", "scene-a", "scene-b", "scene-c")):
        rows.append(
            {
                "pair_id": f"forest:{index}",
                "split": ("train", "development", "test", "train")[index],
                "source_metadata": {"source_scene_group_id": group},
            }
        )
    source = tmp_path / "forest.jsonl"
    output = tmp_path / "proposal"
    _write_jsonl(source, rows)
    subprocess.run(
        [
            sys.executable,
            "scripts/build_qcpr_forest_scene_split_proposal.py",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    audit = json.loads(
        (output / "forest_change_scene_disjoint_split_proposal.json").read_text()
    )
    assert audit["status"] == "SCENE_DISJOINT_SPLIT_PROPOSAL_NOT_RELEASED"
    assert audit["cross_split_component_overlap"] == 0
    proposed = [
        json.loads(line)
        for line in (
            output / "forest_change_pair_manifest_scene_disjoint_proposal.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert proposed[0]["split"] == proposed[1]["split"]
    assert all(row["source_metadata"]["training_enabled"] is False for row in proposed)


def test_tamms_split_proposal_preserves_sequence_groups_and_hold(tmp_path: Path) -> None:
    rows = []
    text = []
    for index in range(8):
        sequence_id = f"tamms:pilot:sequence-{index}"
        frames = [
            {
                "frame_id": f"{sequence_id}:frame:{frame}",
                "timestamp": f"2020-0{frame + 1}-01",
                "sha256": f"{index}-{frame}",
            }
            for frame in range(4)
        ]
        rows.append(
            {
                "sequence_id": sequence_id,
                "frame_count": 4,
                "frames": frames,
                "training_enabled": False,
            }
        )
        text.append(
            {
                "sequence_id": sequence_id,
                "query_id": f"{sequence_id}:q",
                "training_enabled": False,
            }
        )
    source = tmp_path / "tamms.jsonl"
    text_source = tmp_path / "tamms_text.jsonl"
    output = tmp_path / "proposal"
    _write_jsonl(source, rows)
    _write_jsonl(text_source, text)
    subprocess.run(
        [
            sys.executable,
            "scripts/build_qcpr_tamms_split_proposal.py",
            "--manifest",
            str(source),
            "--text-registry",
            str(text_source),
            "--output",
            str(output),
        ],
        check=True,
    )
    audit = json.loads(
        (output / "tamms_long_series_split_proposal_audit.json").read_text()
    )
    assert audit["status"] == "LONG_SERIES_SPLIT_PROPOSAL_NOT_RELEASED"
    assert audit["cross_split_scene_group_overlap"] == 0
    assert audit["image_split_conflicts"] == 0
    assert audit["event_disjoint"] is False
    proposed = [
        json.loads(line)
        for line in (
            output / "tamms_long_series_manifest_split_proposal.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert len(proposed) == 8
    assert {row["split"] for row in proposed} == {"train", "development", "test"}
    assert all(row["training_enabled"] is False for row in proposed)
    text_rows = [
        json.loads(line)
        for line in (
            output / "tamms_long_series_text_registry_split_proposal.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert {row["split"] for row in text_rows} == {row["split"] for row in proposed}
