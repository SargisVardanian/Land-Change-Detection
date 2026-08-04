"""Stable hashes and identifiers for physical assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def item_id(source: str, revision: str, stable_name: str) -> str:
    """Build the canonical ``source:revision:item`` identity."""

    return ":".join(part.replace(":", "_") for part in (source, revision, stable_name))
