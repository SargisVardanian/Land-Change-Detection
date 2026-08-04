#!/usr/bin/env python3
"""Acquire and audit the official RSRCC physical assets without promotion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


REVISION = "7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea"
REPOSITORY = "https://huggingface.co/datasets/google/RSRCC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for path in sorted(root.glob("*_metadata.csv")):
        split = path.name.removesuffix("_metadata.csv")
        count = 0
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                before = str(row.get("before_file_name") or "").strip()
                after = str(row.get("after_file_name") or "").strip()
                if before and after:
                    rows.append({"split": split, "before": before, "after": after})
                count += 1
        counts[split] = count
    return rows, counts


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe RSRCC relative path: {value!r}")
    return path


def download_asset(ref: dict[str, str], repository_root: Path, revision: str) -> dict[str, Any]:
    split = ref["split"]
    relative = safe_relative(ref["relative"])
    target = repository_root / split / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {**ref, "path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target), "downloaded": False}
    remote = f"{split}/{relative.as_posix()}"
    url = f"{REPOSITORY}/resolve/{revision}/{quote(remote, safe='/')}?download=true"
    temporary = target.with_name(target.name + ".part")
    request = Request(url, headers={"User-Agent": "qcpr-rsrcc-audit/1.0"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
                digest.update(block)
                size += len(block)
        os.replace(temporary, target)
        return {**ref, "path": str(target), "bytes": size, "sha256": digest.hexdigest(), "downloaded": True}
    except Exception as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        return {**ref, "path": str(target), "bytes": None, "sha256": None, "downloaded": False, "error": f"{type(exc).__name__}: {exc}"}


def parent_hashes(registry: Path | None) -> set[str]:
    if registry is None or not registry.is_file():
        return set()
    hashes: set[str] = set()
    with registry.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for frame in row.get("frames", []):
                digest = str(frame.get("sha256") or "").strip().casefold()
                if len(digest) == 64:
                    hashes.add(digest)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    metadata_rows, row_counts = read_metadata(args.metadata_root)
    refs: dict[tuple[str, str], dict[str, str]] = {}
    for row in metadata_rows:
        for field in ("before", "after"):
            relative = row[field]
            refs.setdefault((row["split"], relative), {"split": row["split"], "relative": relative})
    records: list[dict[str, Any]] = []
    if args.download and refs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(download_asset, ref, args.repository_root, args.revision) for ref in refs.values()]
            for future in as_completed(futures):
                records.append(future.result())
    else:
        for ref in refs.values():
            path = args.repository_root / ref["split"] / safe_relative(ref["relative"])
            if path.is_file() and path.stat().st_size > 0:
                records.append({**ref, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "downloaded": False})
            else:
                records.append({**ref, "path": str(path), "bytes": None, "sha256": None, "downloaded": False, "error": "MISSING_LOCAL_ASSET"})
    records.sort(key=lambda row: (str(row.get("split")), str(row.get("relative"))))
    missing = [row for row in records if not row.get("sha256")]
    parent = parent_hashes(args.registry)
    shared = sorted({str(row["sha256"]).casefold() for row in records if row.get("sha256") and str(row["sha256"]).casefold() in parent})
    manifest_path = args.output.with_name("rsrcc_physical_asset_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "qcpr-rsrcc-physical-asset-audit-v1",
        "status": "PHYSICAL_ASSETS_ACQUIRED_PARENT_OVERLAP_AUDITED" if not missing else "PHYSICAL_ASSET_ACQUISITION_INCOMPLETE",
        "official_repository": REPOSITORY,
        "revision": args.revision,
        "license_status": "APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED",
        "metadata_row_counts": dict(sorted(row_counts.items())),
        "metadata_row_count": sum(row_counts.values()),
        "unique_physical_pair_filename_tuples": len({(row["split"], row["before"], row["after"]) for row in metadata_rows}),
        "unique_asset_count": len(refs),
        "downloaded_asset_count": sum(bool(row.get("downloaded")) for row in records),
        "available_asset_count": len(records) - len(missing),
        "missing_asset_count": len(missing),
        "download_error_examples": [row.get("error") for row in missing[:20]],
        "parent_registry": str(args.registry) if args.registry else None,
        "parent_frame_hash_count": len(parent),
        "shared_parent_frame_hash_count": len(shared),
        "shared_parent_frame_hash_examples": shared[:20],
        "text_provenance": "generated_language_annotations; evaluation_only; not mask_free_training text",
        "training_enabled": False,
        "asset_manifest": str(manifest_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
