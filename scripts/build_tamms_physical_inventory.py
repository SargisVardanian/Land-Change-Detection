#!/usr/bin/env python3
"""Extract and inventory the physically available TAMMs archive subset.

This command creates only physical sequence records.  TAMMs text remains
generated_unverified and is never promoted by this inventory builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", default="qcpr-tamms-physical-v1")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise ValueError(f"archive member escapes extraction root: {member.name}")
        handle.extractall(root, filter="data")


def split_for(scene_id: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}|{scene_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "development"
    return "test"


def frame_timestamp(path: Path) -> str:
    return datetime.strptime(path.stem, "%Y-%m-%d").date().isoformat()


def build_inventory(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.output.exists():
        raise FileExistsError(args.output)
    archive_sha = sha256_file(args.archive)
    source_revision = f"archive_{archive_sha[:12]}"
    if not args.extract_root.exists() or not any(args.extract_root.iterdir()):
        safe_extract(args.archive, args.extract_root)
    sequence_root = args.extract_root / "waste_disposal"
    sequence_dirs = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise ValueError(f"no sequence directories found under {sequence_root}")

    rows: list[dict[str, Any]] = []
    global_hashes: dict[str, str] = {}
    for sequence_dir in sequence_dirs:
        scene_name = sequence_dir.name
        source_sequence_id = f"waste_disposal/{scene_name}"
        item_id = f"tamms:{source_revision}:{source_sequence_id}"
        image_paths = sorted(sequence_dir.glob("*.jpg"), key=frame_timestamp)
        if len(image_paths) < 3:
            raise ValueError(f"sequence has fewer than three frames: {source_sequence_id}")
        frames = []
        for index, path in enumerate(image_paths):
            digest = sha256_file(path)
            previous = global_hashes.get(digest)
            if previous is not None and previous != str(path):
                raise ValueError(f"duplicate frame hash across sequences: {digest}: {previous} and {path}")
            global_hashes[digest] = str(path)
            frames.append(
                {
                    "frame_id": f"{item_id}:frame:{index}",
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "timestamp": frame_timestamp(path),
                    "sensor": "fMoW-RGB",
                    "gsd": None,
                    "width": None,
                    "height": None,
                }
            )
        split = split_for(source_sequence_id, args.split_seed)
        rows.append(
            {
                "schema_version": "qcpr-tamms-physical-inventory-v1",
                "sequence_id": item_id,
                "source_sequence_id": source_sequence_id,
                "source_version": source_revision,
                "scene_id": f"tamms:{source_sequence_id}",
                "physical_group_id": f"tamms:{source_sequence_id}",
                "event_id": None,
                "frames": frames,
                "split": split,
                "split_policy": "deterministic_sequence_scene_hash_70_15_15",
                "training_enabled": False,
                "quality_status": "PHYSICAL_ONLY_LICENSE_REVIEW_REQUIRED",
                "verification_status": "generated_unverified",
                "provenance": {
                    "archive": str(args.archive),
                    "archive_sha256": archive_sha,
                    "full_official_metadata_not_acquired": True,
                    "generated_text_training_enabled": False,
                },
            }
        )
    metadata_count = None
    if args.metadata and args.metadata.is_file():
        value = json.loads(args.metadata.read_text(encoding="utf-8"))
        metadata_count = len(value) if isinstance(value, list) else None
    audit = {
        "schema_version": "qcpr-tamms-physical-inventory-audit-v1",
        "archive": str(args.archive),
        "archive_sha256": archive_sha,
        "archive_bytes": args.archive.stat().st_size,
        "extract_root": str(args.extract_root.resolve()),
        "sequence_count": len(rows),
        "frame_count": len(global_hashes),
        "duplicate_image_hash_count": 0,
        "metadata_sequence_count": metadata_count,
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "development", "test")},
        "training_enabled": False,
        "text_verification": "generated_unverified",
        "full_official_archive_acquired": False,
    }
    return rows, audit


def main() -> int:
    args = parse_args()
    rows, audit = build_inventory(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda value: value["sequence_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": str(audit_path), **audit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
