from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from land_change_detection.semantic_transitions import (
    changed_area_ratio,
    compute_transition_histogram,
    compute_transition_map,
    dominant_transition,
    load_semantic_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SECOND/SECOND-CC pair-retrieval manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()


def _discover_grouped(root: Path) -> list[dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.stem.lower()
        key = lowered
        for suffix in ("_a", "_b", "_t1", "_t2", "_sem1", "_sem2", "_label", "_mask"):
            if lowered.endswith(suffix):
                key = lowered[: -len(suffix)]
                break
        row = groups.setdefault(key, {})
        full = str(path)
        if lowered.endswith(("_a", "_t1")):
            row["before_path"] = full
        elif lowered.endswith(("_b", "_t2")):
            row["after_path"] = full
        elif lowered.endswith("_sem1"):
            row["semantic_before_path"] = full
        elif lowered.endswith("_sem2"):
            row["semantic_after_path"] = full
        elif lowered.endswith(("_label", "_mask")):
            row["mask_path"] = full
    return [dict(sample_id=key, **value) for key, value in groups.items() if {"before_path", "after_path", "semantic_before_path", "semantic_after_path"} <= set(value)]


def main() -> int:
    args = parse_args()
    discovered = _discover_grouped(args.root)
    rows = []
    distribution: Counter[str] = Counter()
    changed_area_values: list[float] = []
    for row in discovered:
        sem_before = load_semantic_mask(row["semantic_before_path"])
        sem_after = load_semantic_mask(row["semantic_after_path"])
        transition_map = compute_transition_map(sem_before, sem_after, args.num_classes)
        histogram = compute_transition_histogram(transition_map, args.num_classes)
        dominant = dominant_transition(histogram)
        label = f"{dominant[0]}->{dominant[1]}"
        area_ratio = changed_area_ratio(sem_before, sem_after)
        distribution[label] += 1
        changed_area_values.append(area_ratio)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": "SECOND-CC",
                "before_path": row["before_path"],
                "after_path": row["after_path"],
                "semantic_before_path": row["semantic_before_path"],
                "semantic_after_path": row["semantic_after_path"],
                "mask_path": row.get("mask_path"),
                "split": "unknown",
                "transition_histogram": histogram.tolist(),
                "dominant_transition": label,
                "changed_area_ratio": area_ratio,
                "num_classes": args.num_classes,
                "metadata": {
                    "transition_tuple": [int(dominant[0]), int(dominant[1])],
                    "transition_label": label,
                    "curriculum_stage": "stage_3_transition_aware",
                    "retrieval_role": "pair_to_pair_transition_retrieval",
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    diagnostics = {
        "num_rows": len(rows),
        "dataset_name": "SECOND-CC",
        "curriculum_stage": "stage_3_transition_aware",
        "dominant_transition_distribution": dict(distribution),
        "changed_area_ratio_mean": sum(changed_area_values) / max(len(changed_area_values), 1),
        "changed_area_ratio_min": min(changed_area_values) if changed_area_values else 0.0,
        "changed_area_ratio_max": max(changed_area_values) if changed_area_values else 0.0,
        "preview": rows[:5],
    }
    if args.diagnostics_output is not None:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
