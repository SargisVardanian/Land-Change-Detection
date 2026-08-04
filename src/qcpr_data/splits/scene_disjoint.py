"""Scene/component-disjoint split helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


def stable_holdout_split(stable_id: str, *, source_split: str, holdout_fraction: float = 0.2) -> str:
    """Map a source validation partition to development/test deterministically."""

    if source_split == "train":
        return "train"
    if source_split in {"development", "test"}:
        return source_split
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    bucket = int(hashlib.sha256(stable_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < holdout_fraction else "development"


def validate_component_splits(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, set[str]] = {}
    for item in items:
        group = str(item.get("physical_group_id") or item.get("scene_id") or item.get("item_id"))
        by_group.setdefault(group, set()).add(str(item.get("split")))
    leaks = sorted(group for group, splits in by_group.items() if len(splits) > 1)
    return {"component_count": len(by_group), "cross_split_components": leaks, "passed": not leaks}
