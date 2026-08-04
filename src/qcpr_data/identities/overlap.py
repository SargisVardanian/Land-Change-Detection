"""Cross-split overlap and reversed-pair audits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def audit_split_leakage(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    image_splits: dict[str, set[str]] = defaultdict(set)
    pair_signatures: dict[tuple[str, ...], set[str]] = defaultdict(set)
    scene_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in items:
        split = str(item.get("split"))
        source = str(item.get("source"))
        scene = str(item.get("scene_id"))
        scene_splits[(source, scene)].add(split)
        frame_hashes = [str(frame.get("sha256") or frame.get("path")) for frame in item.get("frames", [])]
        for digest in frame_hashes:
            image_splits[digest].add(split)
        if len(frame_hashes) == 2:
            pair_signatures[tuple(sorted(frame_hashes))].add(split)
    shared_images = sorted(key for key, splits in image_splits.items() if len(splits) > 1)
    reversed_pairs = sorted(key for key, splits in pair_signatures.items() if len(splits) > 1)
    scene_leaks = sorted(f"{source}:{scene}" for (source, scene), splits in scene_splits.items() if len(splits) > 1)
    return {
        "shared_image_count": len(shared_images),
        "shared_image_examples": shared_images[:20],
        "reversed_pair_count": len(reversed_pairs),
        "reversed_pair_examples": [list(pair) for pair in reversed_pairs[:20]],
        "scene_split_count": len(scene_leaks),
        "scene_split_examples": scene_leaks[:20],
        "passed": not shared_images and not reversed_pairs and not scene_leaks,
    }
