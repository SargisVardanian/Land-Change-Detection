"""Sequence overlap validator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def validate_sequence_disjoint(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    frame_splits: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.get("item_type") != "sequence":
            continue
        for frame in item.get("frames", []):
            frame_splits[str(frame.get("sha256") or frame.get("path"))].add(str(item.get("split")))
    leaks = sorted(frame for frame, splits in frame_splits.items() if len(splits) > 1)
    return {"shared_frame_count": len(leaks), "shared_frame_examples": leaks[:20], "passed": not leaks}
