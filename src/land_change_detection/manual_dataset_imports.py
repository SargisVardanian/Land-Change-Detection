from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from land_change_detection.change_retrieval_datasets import ChangeRetrievalSample, discover_change_samples
from land_change_detection.semantic_transitions import (
    changed_area_ratio,
    compute_transition_histogram,
    compute_transition_map,
    dominant_transition,
    load_semantic_mask,
)


def directory_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "preview_files": [str(path.relative_to(root)) for path in files[:20]],
    }


def ensure_symlink(link_path: Path, target: Path) -> Path:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return link_path
        link_path.unlink()
    elif link_path.exists():
        return link_path
    link_path.symlink_to(target)
    return link_path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def samples_to_rows(samples: list[ChangeRetrievalSample]) -> list[dict[str, Any]]:
    return [sample.to_dict() for sample in samples]


def build_transition_manifest_rows(
    samples: list[ChangeRetrievalSample],
    *,
    dataset_name: str,
    num_classes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.semantic_before_path or not sample.semantic_after_path:
            continue
        sem_before = load_semantic_mask(sample.semantic_before_path)
        sem_after = load_semantic_mask(sample.semantic_after_path)
        transition_map = compute_transition_map(sem_before, sem_after, num_classes)
        histogram = compute_transition_histogram(transition_map, num_classes)
        dominant = dominant_transition(histogram)
        label = f"{dominant[0]}->{dominant[1]}"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset_name": dataset_name,
                "before_path": sample.before_path,
                "after_path": sample.after_path,
                "semantic_before_path": sample.semantic_before_path,
                "semantic_after_path": sample.semantic_after_path,
                "mask_path": sample.mask_path,
                "split": sample.split or "unknown",
                "transition_histogram": histogram.tolist(),
                "dominant_transition": label,
                "changed_area_ratio": changed_area_ratio(sem_before, sem_after),
                "num_classes": num_classes,
                "metadata": {
                    **dict(sample.metadata),
                    "transition_tuple": [int(dominant[0]), int(dominant[1])],
                    "transition_label": label,
                    "curriculum_stage": "stage_3_transition_aware",
                    "retrieval_role": "pair_to_pair_transition_retrieval",
                },
            }
        )
    return rows


def canonical_import_outputs(project_root: Path, dataset_slug: str) -> dict[str, Path]:
    indexes_root = project_root / "indexes"
    return {
        "sample_manifest": indexes_root / f"manual_{dataset_slug}_samples.jsonl",
        "pair_manifest": indexes_root / f"manual_{dataset_slug}_pair_manifest.jsonl",
        "report": indexes_root / f"manual_{dataset_slug}_import_report.json",
    }


def build_generic_change_sample_manifest(root: Path, dataset_name: str) -> list[ChangeRetrievalSample]:
    return discover_change_samples(root, dataset_name)


def build_grouped_semantic_samples(root: Path, dataset_name: str) -> list[ChangeRetrievalSample]:
    groups: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.stem.lower()
        key = lowered
        for suffix in ("_a", "_b", "_t1", "_t2", "_before", "_after", "_sem1", "_sem2", "_label", "_mask"):
            if lowered.endswith(suffix):
                key = lowered[: -len(suffix)]
                break
        row = groups.setdefault(key, {})
        full = str(path)
        if lowered.endswith(("_a", "_t1", "_before")):
            row["before_path"] = full
        elif lowered.endswith(("_b", "_t2", "_after")):
            row["after_path"] = full
        elif lowered.endswith("_sem1"):
            row["semantic_before_path"] = full
        elif lowered.endswith("_sem2"):
            row["semantic_after_path"] = full
        elif lowered.endswith(("_label", "_mask")):
            row["mask_path"] = full
    samples: list[ChangeRetrievalSample] = []
    for key, row in sorted(groups.items()):
        if {"before_path", "after_path"} - set(row):
            continue
        samples.append(
            ChangeRetrievalSample(
                sample_id=key,
                dataset_name=dataset_name,
                before_path=row["before_path"],
                after_path=row["after_path"],
                mask_path=row.get("mask_path"),
                semantic_before_path=row.get("semantic_before_path"),
                semantic_after_path=row.get("semantic_after_path"),
                split=None,
                metadata={"source_root": str(root)},
            )
        )
    return samples
