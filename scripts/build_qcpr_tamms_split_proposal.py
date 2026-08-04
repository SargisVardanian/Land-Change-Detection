#!/usr/bin/env python3
"""Build a deterministic, unreleased split proposal for the TAMMs pilot.

TAMMs supplies temporal sequences but the pilot metadata has no official split
or event identity. Sequence IDs are therefore used only as scene groups; this
script never enables training or claims event-disjoint generalisation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def scene_group(row: dict[str, Any]) -> str:
    value = row.get("source_scene_group_id") or row.get("sequence_id")
    if not value:
        raise SystemExit("TAMMs row has no sequence or scene-group identity")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-registry", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit("TAMMs split proposal requires at least one sequence")

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    image_to_group: dict[str, str] = {}
    for row in rows:
        if int(row.get("frame_count", 0)) != 4:
            raise SystemExit("TAMMs split proposal requires four frames per sequence")
        frames = row.get("frames") or []
        if len(frames) != 4 or len({frame.get("frame_id") for frame in frames}) != 4:
            raise SystemExit("TAMMs sequence has invalid frame identity")
        timestamps = [str(frame.get("timestamp")) for frame in frames]
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise SystemExit("TAMMs timestamps are not strictly ordered")
        group_id = scene_group(row)
        groups[group_id].append(row)
        for frame in frames:
            image_hash = frame.get("sha256")
            if image_hash:
                previous = image_to_group.setdefault(str(image_hash), group_id)
                if previous != group_id:
                    raise SystemExit("an image hash is shared by two scene groups")
        if row.get("training_enabled") is not False:
            raise SystemExit("unverified TAMMs rows must remain training_enabled=false")

    total = len(rows)
    targets = {
        "train": round(total * 0.70),
        "development": round(total * 0.15),
        "test": total - round(total * 0.70) - round(total * 0.15),
    }
    assigned: collections.Counter[str] = collections.Counter()
    group_split: dict[str, str] = {}
    ordered_groups = sorted(
        groups.items(), key=lambda item: (-len(item[1]), stable_hash(item[0]))
    )
    for group_id, group_rows in ordered_groups:
        split = max(
            targets,
            key=lambda name: (targets[name] - assigned[name], name == "train", name),
        )
        group_split[group_id] = split
        assigned[split] += len(group_rows)

    proposal: list[dict[str, Any]] = []
    for row in rows:
        group_id = scene_group(row)
        updated = dict(row)
        updated["split"] = group_split[group_id]
        updated["source_scene_group_id"] = group_id
        updated["source_event_id"] = row.get("source_event_id")
        updated["training_enabled"] = False
        updated["split_policy"] = "sequence_scene_disjoint_proposal_without_official_event_split"
        proposal.append(updated)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "tamms_long_series_manifest_split_proposal.jsonl"
    write_jsonl(manifest_path, proposal)

    text_rows: list[dict[str, Any]] = []
    if args.text_registry:
        for row in read_jsonl(args.text_registry):
            sequence_id = str(row.get("sequence_id", ""))
            if sequence_id not in group_split:
                raise SystemExit(f"text row references unknown TAMMs sequence {sequence_id}")
            updated = dict(row)
            updated["split"] = group_split[sequence_id]
            updated["source_scene_group_id"] = sequence_id
            updated["training_enabled"] = False
            text_rows.append(updated)
        write_jsonl(
            args.output / "tamms_long_series_text_registry_split_proposal.jsonl",
            text_rows,
        )

    groups_by_split: dict[str, set[str]] = collections.defaultdict(set)
    for group_id, split in group_split.items():
        groups_by_split[split].add(group_id)
    overlap = sum(
        len(groups_by_split[left] & groups_by_split[right])
        for left in groups_by_split
        for right in groups_by_split
        if left < right
    )
    event_ids = {
        str(row.get("source_event_id")) for row in rows if row.get("source_event_id")
    }
    image_split: dict[str, str] = {}
    image_split_conflicts = 0
    for row in proposal:
        for frame in row["frames"]:
            image_hash = frame.get("sha256")
            if not image_hash:
                continue
            previous = image_split.setdefault(str(image_hash), str(row["split"]))
            if previous != row["split"]:
                image_split_conflicts += 1

    audit = {
        "status": "LONG_SERIES_SPLIT_PROPOSAL_NOT_RELEASED",
        "source": "TAMMs",
        "input_manifest": str(args.manifest),
        "sequence_count": len(proposal),
        "text_count": len(text_rows),
        "scene_group_count": len(groups),
        "target_sequence_counts": targets,
        "proposed_sequence_counts": dict(sorted(assigned.items())),
        "proposed_scene_group_counts": {
            split: len(group_ids) for split, group_ids in sorted(groups_by_split.items())
        },
        "cross_split_scene_group_overlap": overlap,
        "image_split_conflicts": image_split_conflicts,
        "source_event_ids_present": len(event_ids),
        "event_disjoint": False,
        "official_split_present": False,
        "training_enabled": False,
        "reason": "TAMMs pilot has no official split or event identity; proposal is sequence-disjoint only and requires source-level review before any release",
        "required_followup": [
            "obtain or define authoritative event/scene split metadata",
            "independently verify generated text",
            "annotate change onset, duration and relevant frame range",
        ],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    write_json(args.output / "tamms_long_series_split_proposal_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
