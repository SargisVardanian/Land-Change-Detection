from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from PIL import Image

from land_change_detection.models.qcpr_v3_losses import query_mask_metrics


@dataclass(frozen=True)
class MaskObjective:
    name: str
    dice_weight: float
    focal_weight: float
    tversky_weight: float
    positive_focal_weight: float
    negative_focal_weight: float


MASK_OBJECTIVES = {
    "A": MaskObjective("A_dice_balanced_focal", 1.0, 1.0, 0.0, 0.75, 0.25),
}


def resolve_mask_objective(name: str) -> MaskObjective:
    try:
        return MASK_OBJECTIVES[name]
    except KeyError as exc:
        raise ValueError(f"unknown mask objective {name!r}; expected one of {sorted(MASK_OBJECTIVES)}") from exc


def exact_s2looking_query_indices(dataset: Dataset) -> list[int]:
    rows = getattr(dataset, "samples", None)
    if rows is None:
        raise TypeError("exact S2Looking filtering requires a manifest dataset with .samples")
    selected: list[int] = []
    for index, row in enumerate(rows):
        metadata = row.get("source_metadata", {})
        dataset_name = str(row.get("dataset_name", "")).casefold()
        mode = str(row.get("seg_supervision_mode", metadata.get("seg_supervision_mode", ""))).casefold()
        retrieval = bool(row.get("retrieval_supervision", metadata.get("retrieval_supervision", True)))
        if dataset_name == "s2looking" and mode == "query_specific" and not retrieval:
            selected.append(index)
    if not selected:
        raise RuntimeError("exact S2Looking query-specific localization filter selected zero rows")
    return selected


def _mask_area_fraction(path: str | Path) -> float:
    with Image.open(path) as image:
        binary = image.convert("L").point(lambda value: 255 if value > 0 else 0)
        if binary.getbbox() is None:
            return 0.0
        histogram = binary.histogram()
        foreground = sum(histogram[1:])
        return foreground / max(image.width * image.height, 1)


def fixed_probe_subset(dataset: Dataset, *, count: int, pair_ids: tuple[str, ...] = ()) -> Subset:
    if count <= 0:
        raise ValueError("fixed probe count must be positive")
    if count % 2:
        raise ValueError("fixed S2Looking probe count must be even to preserve appeared/disappeared pairs")
    rows = getattr(dataset, "samples", None)
    if rows is None:
        raise TypeError("fixed S2Looking probes require a manifest dataset")
    if pair_ids:
        by_id = {str(row["pair_id"]): index for index, row in enumerate(rows)}
        missing = [pair_id for pair_id in pair_ids if pair_id not in by_id]
        if missing:
            raise ValueError(f"fixed probe pair IDs are absent: {missing}")
        selected = [by_id[pair_id] for pair_id in pair_ids]
        directions = {str(rows[index]["pair_id"]).rsplit(":", 1)[-1] for index in selected}
        if directions != {"appeared", "disappeared"}:
            raise ValueError("explicit fixed probe must include appeared and disappeared directions")
        return Subset(dataset, selected)
    grouped: dict[str, dict[str, int]] = {}
    for index in exact_s2looking_query_indices(dataset):
        row = rows[index]
        change = str(row.get("source_metadata", {}).get("change_type", row.get("change_type", "")))
        base = str(row["pair_id"]).rsplit(":", 1)[0]
        if change in {"appeared", "disappeared"}:
            grouped.setdefault(base, {})[change] = index
    indices: list[int] = []
    for base in sorted(grouped):
        directions = grouped[base]
        if set(directions) != {"appeared", "disappeared"}:
            continue
        pair_indices = [directions["appeared"], directions["disappeared"]]
        areas = [_mask_area_fraction(rows[index]["mask_path"]) for index in pair_indices]
        if max(areas) <= 0.0:
            continue
        indices.extend(pair_indices)
        if len(indices) >= count:
            break
    if len(indices) < count:
        raise RuntimeError(f"requested {count} non-trivial fixed S2Looking rows, found {len(indices)}")
    return Subset(dataset, indices[:count])


def subset_pair_ids(subset: Subset) -> list[str]:
    dataset = subset.dataset
    rows = getattr(dataset, "samples", None)
    if rows is None:
        raise TypeError("pair IDs require a manifest dataset")
    return [str(rows[int(index)]["pair_id"]) for index in subset.indices]


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_probe_contract(
    *, train_subset: Subset, validation_subset: Subset,
    train_manifest: str | Path, validation_manifest: str | Path,
) -> dict[str, Any]:
    train_ids = subset_pair_ids(train_subset)
    validation_ids = subset_pair_ids(validation_subset)
    train_rows = getattr(train_subset.dataset, "samples")
    validation_rows = getattr(validation_subset.dataset, "samples")
    all_train_ids = [str(train_rows[index]["pair_id"]) for index in exact_s2looking_query_indices(train_subset.dataset)]
    all_validation_ids = [str(validation_rows[index]["pair_id"]) for index in exact_s2looking_query_indices(validation_subset.dataset)]
    train_areas = [_mask_area_fraction(train_rows[int(index)]["mask_path"]) for index in train_subset.indices]
    validation_areas = [_mask_area_fraction(validation_rows[int(index)]["mask_path"]) for index in validation_subset.indices]
    if not any(area > 0 for area in train_areas) or not any(area > 0 for area in validation_areas):
        raise RuntimeError("fixed probes must contain non-empty directional masks")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("fixed train and validation probes overlap by pair ID")
    return {
        "filter": {
            "dataset_name": "s2looking",
            "seg_supervision_mode": "query_specific",
            "retrieval_supervision": False,
        },
        "train_pair_ids": train_ids,
        "validation_pair_ids": validation_ids,
        "train_target_area_fractions": train_areas,
        "validation_target_area_fractions": validation_areas,
        "train_nonempty_count": sum(area > 0 for area in train_areas),
        "validation_nonempty_count": sum(area > 0 for area in validation_areas),
        "all_filtered_train_pair_ids": all_train_ids,
        "all_filtered_validation_pair_ids": all_validation_ids,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "train_manifest_sha256": manifest_sha256(train_manifest),
        "validation_manifest_sha256": manifest_sha256(validation_manifest),
    }


def swapped_direction_indices(pair_ids: Sequence[str], change_types: Sequence[str | None]) -> list[int]:
    if len(pair_ids) != len(change_types):
        raise ValueError("pair IDs and change types must align")
    lookup: dict[tuple[str, str], int] = {}
    for index, (pair_id, change) in enumerate(zip(pair_ids, change_types, strict=True)):
        if change not in {"appeared", "disappeared"}:
            continue
        base = pair_id.rsplit(":", 1)[0]
        lookup[(base, str(change))] = index
    result: list[int] = []
    for pair_id, change in zip(pair_ids, change_types, strict=True):
        opposite = "disappeared" if change == "appeared" else "appeared"
        result.append(lookup.get((pair_id.rsplit(":", 1)[0], opposite), -1))
    return result


def query_swap_metrics(correct_logits: Tensor, swapped_logits: Tensor, targets: Tensor) -> dict[str, float]:
    if correct_logits.shape != swapped_logits.shape or targets.shape != correct_logits.shape:
        raise ValueError("correct, swapped and target masks must align")
    correct = query_mask_metrics(correct_logits, targets)
    swapped = query_mask_metrics(swapped_logits, targets)
    correct_iou = float(correct["nonempty_iou"])
    swapped_iou = float(swapped["nonempty_iou"])
    correct_soft_iou = float(correct["nonempty_soft_iou"])
    swapped_soft_iou = float(swapped["nonempty_soft_iou"])
    return {
        "correct_query_iou": correct_iou,
        "swapped_query_iou": swapped_iou,
        "query_swap_gap": correct_iou - swapped_iou,
        "correct_query_soft_iou": correct_soft_iou,
        "swapped_query_soft_iou": swapped_soft_iou,
        "soft_query_swap_iou_gap": correct_soft_iou - swapped_soft_iou,
    }


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
