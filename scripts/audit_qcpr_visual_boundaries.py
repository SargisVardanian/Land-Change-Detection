#!/usr/bin/env python3
"""Audit perceptual near-duplicates across source and split boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def dhash(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def query(self, value: int, radius: int) -> Iterable[tuple[int, int]]:
        if self.root is None:
            return
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[0], distance
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node[1].items() if low <= edge <= high)


def _cross_count(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str) -> int:
    if left is right:
        total = len(left) * (len(left) - 1) // 2
        same = sum(value * (value - 1) // 2 for value in Counter(row[key] for row in left).values())
        return total - same
    total = len(left) * len(right)
    left_counts = Counter(row[key] for row in left)
    right_counts = Counter(row[key] for row in right)
    return total - sum(count * right_counts[value] for value, count in left_counts.items())


def _different_item_count(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    if left is right:
        total = len(left) * (len(left) - 1) // 2
        same = sum(value * (value - 1) // 2 for value in Counter(row["item_id"] for row in left).values())
        return total - same
    total = len(left) * len(right)
    right_counts = Counter(row["item_id"] for row in right)
    return total - sum(right_counts[row["item_id"]] for row in left)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--example-limit", type=int, default=200)
    args = parser.parse_args()
    try:
        args.output.resolve().relative_to(args.release.resolve())
    except ValueError:
        pass
    else:
        parser.error("--output must be outside immutable --release")

    registry = args.release / "registries/frames.jsonl"
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    frame_count = 0
    with registry.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            frame_count += 1
            path = Path(str(row.get("native_path") or row.get("path")))
            try:
                value = dhash(path)
            except Exception as exc:  # pragma: no cover - real-data evidence
                failures.append(
                    {"line": str(line_number), "path": str(path), "error": type(exc).__name__}
                )
                continue
            groups[value].append(
                {
                    "frame_id": str(row.get("frame_id")),
                    "item_id": str(row.get("physical_item_id") or row.get("item_id")),
                    "source": str(row.get("source")),
                    "split": str(row.get("split")),
                    "sha256": str(row.get("native_sha256") or row.get("sha256")),
                    "path": str(path),
                }
            )

    counters: Counter[str] = Counter()
    by_distance: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    tree = BKTree()
    for value in sorted(groups):
        right = groups[value]
        # Same-hash candidates are evaluated once within the bucket.
        candidate_count = _different_item_count(right, right)
        if candidate_count:
            counters["different_item_pairs"] += candidate_count
            counters["cross_source_pairs"] += _cross_count(right, right, "source")
            counters["cross_split_pairs"] += _cross_count(right, right, "split")
            by_distance[0] += candidate_count
            by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in right:
                by_sha[row["sha256"]].append(row)
            counters["exact_asset_cross_source_pairs"] += sum(
                _cross_count(rows, rows, "source") for rows in by_sha.values()
            )
            counters["exact_asset_cross_split_pairs"] += sum(
                _cross_count(rows, rows, "split") for rows in by_sha.values()
            )
            if len(examples) < args.example_limit:
                for index, a in enumerate(right):
                    for b in right[index + 1 :]:
                        if a["item_id"] == b["item_id"]:
                            continue
                        if a["source"] == b["source"] and a["split"] == b["split"]:
                            continue
                        examples.append(
                            {
                                "distance": 0,
                                "boundary": {
                                    "cross_source": a["source"] != b["source"],
                                    "cross_split": a["split"] != b["split"],
                                },
                                "classification": (
                                    "exact_asset_duplicate"
                                    if a["sha256"] == b["sha256"]
                                    else "perceptual_identical_candidate"
                                ),
                                "left": a,
                                "right": b,
                            }
                        )
                        if len(examples) >= args.example_limit:
                            break
                    if len(examples) >= args.example_limit:
                        break
        for other_value, distance in tree.query(value, args.threshold):
            left = groups[other_value]
            candidate_count = _different_item_count(left, right)
            if not candidate_count:
                continue
            counters["different_item_pairs"] += candidate_count
            counters["cross_source_pairs"] += _cross_count(left, right, "source")
            counters["cross_split_pairs"] += _cross_count(left, right, "split")
            by_distance[distance] += candidate_count
            if len(examples) < args.example_limit:
                for a in left:
                    for b in right:
                        if a["item_id"] == b["item_id"]:
                            continue
                        if a["source"] == b["source"] and a["split"] == b["split"]:
                            continue
                        examples.append(
                            {
                                "distance": distance,
                                "boundary": {
                                    "cross_source": a["source"] != b["source"],
                                    "cross_split": a["split"] != b["split"],
                                },
                                "classification": (
                                    "exact_asset_duplicate"
                                    if a["sha256"] == b["sha256"]
                                    else "perceptual_identical_candidate"
                                    if distance == 0
                                    else "visual_near_duplicate_candidate"
                                ),
                                "left": a,
                                "right": b,
                            }
                        )
                        if len(examples) >= args.example_limit:
                            break
                    if len(examples) >= args.example_limit:
                        break
        tree.add(value)

    result = {
        "schema_version": "qcpr-visual-boundary-audit-v1",
        "release": str(args.release),
        "algorithm": "64-bit dHash over grayscale 9x8 native-frame resize",
        "hamming_threshold": args.threshold,
        "comparison_scope": "all sources and splits; same physical item excluded",
        "frame_registry_sha256": _sha256(registry),
        "frame_count": frame_count,
        "decoded_frame_count": frame_count - len(failures),
        "decode_failure_count": len(failures),
        "unique_dhash_count": len(groups),
        "candidate_pair_counts": dict(counters),
        "candidate_pairs_by_hamming_distance": {
            str(key): value for key, value in sorted(by_distance.items())
        },
        "examples_are_review_candidates_not_confirmed_leakage": True,
        "examples": examples,
        "decode_failures": failures[: args.example_limit],
        "passed_decode_gate": not failures and frame_count == 66742,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("frame_count", "decoded_frame_count", "unique_dhash_count", "candidate_pair_counts", "passed_decode_gate")}, sort_keys=True))
    return 0 if result["passed_decode_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
