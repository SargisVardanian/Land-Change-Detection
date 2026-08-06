#!/usr/bin/env python3
"""Acquire the physical RSRCC image assets from the pinned official HF revision.

The revision is obtained from an official Dataset Viewer asset URL and must be
passed explicitly.  This path avoids re-fetching every Viewer row after the
metadata/order audit while keeping RSRCC generated text evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from acquire_rsrcc_via_dataset_viewer import (
    acquire_assets,
    read_metadata_rows,
    safe_relative,
    sha256_file,
)


DATASET = "google/RSRCC"
HF_RESOLVE_BASE = "https://huggingface.co/datasets/google/RSRCC/resolve"
SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}


def build_raw_entries(
    metadata: Mapping[str, list[Mapping[str, str]]],
    *,
    revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assets: dict[tuple[str, str], dict[str, Any]] = {}
    row_counts = {split: len(rows) for split, rows in metadata.items()}
    for local_split, rows in metadata.items():
        source_split = SPLIT_MAP[local_split]
        for row_index, row in enumerate(rows):
            for side in ("before", "after"):
                relative = str(row[side])
                safe_relative(relative)
                key = (local_split, relative)
                entry = assets.get(key)
                if entry is None:
                    entry = {
                        "dataset": DATASET,
                        "revision": revision,
                        "local_split": local_split,
                        "source_split": source_split,
                        "relative": relative,
                        "side": side,
                        "first_metadata_row_index": row_index,
                        "reference_count": 0,
                        "asset_url": (
                            f"{HF_RESOLVE_BASE}/{quote(revision, safe='')}/"
                            f"{quote(source_split, safe='')}/{quote(relative, safe='/')}"
                        ),
                    }
                    assets[key] = entry
                entry["reference_count"] += 1
    audit = {
        "dataset": DATASET,
        "revision": revision,
        "metadata_row_counts": row_counts,
        "metadata_row_count": sum(row_counts.values()),
        "source_split_map": dict(SPLIT_MAP),
        "text_provenance": "RSRCC generated language annotations; evaluation-only",
        "text_training_enabled": False,
    }
    return (
        sorted(
            assets.values(),
            key=lambda row: (str(row["local_split"]), str(row["relative"]), str(row["side"])),
        ),
        audit,
    )


def resolve_existing(entries: list[dict[str, Any]], repository_root: Path) -> list[dict[str, Any]]:
    resolved = []
    for original in entries:
        entry = dict(original)
        target = repository_root / str(entry["local_split"]) / safe_relative(str(entry["relative"]))
        entry["path"] = str(target)
        if target.is_file() and target.stat().st_size > 0:
            entry.update({
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "downloaded": False,
                "acquisition": "preexisting",
            })
        else:
            entry.update({
                "bytes": None,
                "sha256": None,
                "downloaded": False,
                "error": "MISSING_LOCAL_ASSET",
            })
        resolved.append(entry)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    metadata = read_metadata_rows(args.metadata_root)
    entries, audit = build_raw_entries(metadata, revision=args.revision)
    if args.download:
        entries = acquire_assets(entries, args.repository_root, args.workers)
    else:
        entries = resolve_existing(entries, args.repository_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    missing = [row for row in entries if not row.get("sha256")]
    report = {
        "schema_version": "qcpr-rsrcc-hf-raw-physical-acquisition-v1",
        "status": (
            "PHYSICAL_ASSET_ACQUISITION_COMPLETE"
            if args.download and not missing
            else "PHYSICAL_ASSET_ACQUISITION_INCOMPLETE"
            if args.download
            else "PHYSICAL_MANIFEST_RESOLVED_DOWNLOAD_PENDING"
        ),
        "dataset": DATASET,
        "revision": args.revision,
        "raw_resolve_base": HF_RESOLVE_BASE,
        "metadata_root": str(args.metadata_root),
        "repository_root": str(args.repository_root),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "download_mode": bool(args.download),
        "workers": args.workers,
        "unique_asset_count": len(entries),
        "available_asset_count": sum(bool(row.get("sha256")) for row in entries),
        "missing_asset_count": len(missing),
        "downloaded_asset_count": sum(bool(row.get("downloaded")) for row in entries),
        "preexisting_asset_count": sum(row.get("acquisition") == "preexisting" for row in entries),
        "reference_count_total": sum(int(row.get("reference_count") or 0) for row in entries),
        "metadata_row_counts": audit["metadata_row_counts"],
        "metadata_row_count": audit["metadata_row_count"],
        "source_split_map": audit["source_split_map"],
        "text_provenance": audit["text_provenance"],
        "text_training_enabled": False,
        "source_revision_evidence": "Pinned from official Dataset Viewer cached-assets URL; raw paths verified by direct official HF resolve probes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not missing else (2 if args.download else 0)


if __name__ == "__main__":
    raise SystemExit(main())
