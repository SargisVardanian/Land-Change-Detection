"""RSRCC physical adapter for audited assets.

Generated RSRCC language annotations are never enabled by this adapter.  The
physical manifest loader only validates paths and hashes supplied by the
acquisition audit; release policy and licensing remain outside the adapter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from .common import RegistrySourceAdapter, iter_jsonl


ADAPTER = RegistrySourceAdapter("RSRCC", "physical-manifest-hold")
PHYSICAL_MANIFEST_SCHEMA = "qcpr-rsrcc-hf-raw-physical-acquisition-v1"


def iter_physical_assets(manifest_path: Path) -> Iterator[dict[str, Any]]:
    """Yield acquired physical asset rows without enabling text supervision."""
    for row in iter_jsonl(manifest_path):
        if str(row.get("dataset") or "").casefold() != "google/rsrcc":
            continue
        yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_physical_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate manifest paths/hashes; this does not validate generated text."""
    rows = list(iter_physical_assets(manifest_path))
    missing = []
    mismatches = []
    for row in rows:
        relative = str(row.get("relative") or "")
        local_split = str(row.get("local_split") or "")
        path = repository_root / local_split / relative
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append({"path": str(path), "asset_id": relative})
            continue
        expected = str(row.get("sha256") or "")
        if verify_hashes and expected and _sha256(path) != expected:
            mismatches.append({"path": str(path), "expected": expected, "actual": _sha256(path)})
    return {
        "schema_version": PHYSICAL_MANIFEST_SCHEMA,
        "manifest": str(manifest_path),
        "repository_root": str(repository_root),
        "asset_count": len(rows),
        "missing_count": len(missing),
        "hash_mismatch_count": len(mismatches),
        "missing": missing,
        "hash_mismatches": mismatches,
        "passed": bool(rows) and not missing and not mismatches,
        "training_enabled": False,
        "text_training_enabled": False,
    }


def iter_items(*args, **kwargs):
    return ADAPTER.pairs(*args, **kwargs)
