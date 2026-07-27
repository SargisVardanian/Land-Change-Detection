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
    parser = argparse.ArgumentParser(description="Build a Hi-UCD pair-retrieval manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-dir", type=Path, default=None)
    parser.add_argument("--after-dir", type=Path, default=None)
    parser.add_argument("--semantic-before-dir", type=Path, default=None)
    parser.add_argument("--semantic-after-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()


def _index_dir(path: Path | None) -> dict[str, Path]:
    if path is None or not path.exists():
        return {}
    return {item.stem.lower(): item for item in sorted(path.rglob("*")) if item.is_file()}


def _recursive_discover(root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict[str, Path]]:
    before: dict[str, Path] = {}
    after: dict[str, Path] = {}
    sem_before: dict[str, Path] = {}
    sem_after: dict[str, Path] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        lowered = item.stem.lower()
        if lowered.endswith(("_a", "_t1", "_before")):
            before[lowered.rsplit("_", 1)[0]] = item
        elif lowered.endswith(("_b", "_t2", "_after")):
            after[lowered.rsplit("_", 1)[0]] = item
        elif lowered.endswith(("_sem1", "_label1", "_semanticbefore")):
            sem_before[lowered.rsplit("_", 1)[0]] = item
        elif lowered.endswith(("_sem2", "_label2", "_semanticafter")):
            sem_after[lowered.rsplit("_", 1)[0]] = item
    return before, after, sem_before, sem_after


def main() -> int:
    args = parse_args()
    if all(path is not None for path in (args.before_dir, args.after_dir, args.semantic_before_dir, args.semantic_after_dir)):
        before = _index_dir(args.before_dir)
        after = _index_dir(args.after_dir)
        sem_before = _index_dir(args.semantic_before_dir)
        sem_after = _index_dir(args.semantic_after_dir)
    else:
        before, after, sem_before, sem_after = _recursive_discover(args.root)

    shared = sorted(set(before) & set(after) & set(sem_before) & set(sem_after))
    if not shared:
        diagnostics = {
            "error": "No matched Hi-UCD samples discovered.",
            "before_candidates": len(before),
            "after_candidates": len(after),
            "semantic_before_candidates": len(sem_before),
            "semantic_after_candidates": len(sem_after),
        }
        if args.diagnostics_output is not None:
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(json.dumps(diagnostics, indent=2))

    rows = []
    distribution: Counter[str] = Counter()
    changed_area_values: list[float] = []
    for key in shared:
        sem_b = load_semantic_mask(sem_before[key])
        sem_a = load_semantic_mask(sem_after[key])
        transition_map = compute_transition_map(sem_b, sem_a, args.num_classes)
        histogram = compute_transition_histogram(transition_map, args.num_classes)
        dominant = dominant_transition(histogram)
        label = f"{dominant[0]}->{dominant[1]}"
        area_ratio = changed_area_ratio(sem_b, sem_a)
        distribution[label] += 1
        changed_area_values.append(area_ratio)
        rows.append(
            {
                "sample_id": key,
                "dataset_name": "Hi-UCD",
                "before_path": str(before[key]),
                "after_path": str(after[key]),
                "semantic_before_path": str(sem_before[key]),
                "semantic_after_path": str(sem_after[key]),
                "mask_path": None,
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
        "dataset_name": "Hi-UCD",
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
