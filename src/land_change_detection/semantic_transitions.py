from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image


def load_semantic_mask(
    path: str | Path,
    color_map: Mapping[tuple[int, int, int], int] | None = None,
    ignore_index: int | None = None,
) -> np.ndarray:
    image = Image.open(path)
    if color_map is None:
        array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., 0]
        return array.astype(np.int64)

    rgb = np.asarray(image.convert("RGB"))
    mask = np.full(rgb.shape[:2], ignore_index if ignore_index is not None else 0, dtype=np.int64)
    for color, class_index in color_map.items():
        matches = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        mask[matches] = int(class_index)
    return mask


def compute_transition_map(semantic_before: np.ndarray, semantic_after: np.ndarray, num_classes: int) -> np.ndarray:
    if semantic_before.shape != semantic_after.shape:
        raise ValueError(f"semantic shapes must match: {semantic_before.shape} vs {semantic_after.shape}")
    return semantic_before.astype(np.int64) * int(num_classes) + semantic_after.astype(np.int64)


def compute_transition_histogram(
    transition_map: np.ndarray,
    num_classes: int,
    ignore_no_change: bool = True,
    ignore_index: int | None = None,
) -> np.ndarray:
    flat = transition_map.reshape(-1)
    if ignore_index is not None:
        ignore_pairs = {
            ignore_index * num_classes + class_index
            for class_index in range(num_classes)
        } | {
            class_index * num_classes + ignore_index
            for class_index in range(num_classes)
        }
        flat = flat[~np.isin(flat, list(ignore_pairs))]
    if ignore_no_change:
        no_change = np.asarray([class_index * num_classes + class_index for class_index in range(num_classes)], dtype=np.int64)
        flat = flat[~np.isin(flat, no_change)]
    counts = np.bincount(flat, minlength=num_classes * num_classes).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def dominant_transition(transition_histogram: np.ndarray) -> tuple[int, int, float]:
    if transition_histogram.size == 0:
        return (0, 0, 0.0)
    index = int(np.argmax(transition_histogram))
    size = int(round(np.sqrt(transition_histogram.size))) or 1
    return (index // size, index % size, float(transition_histogram[index]))


def changed_area_ratio(
    semantic_before: np.ndarray,
    semantic_after: np.ndarray,
    ignore_index: int | None = None,
) -> float:
    if semantic_before.shape != semantic_after.shape:
        raise ValueError(f"semantic shapes must match: {semantic_before.shape} vs {semantic_after.shape}")
    valid = np.ones_like(semantic_before, dtype=bool)
    if ignore_index is not None:
        valid &= semantic_before != ignore_index
        valid &= semantic_after != ignore_index
    denom = int(valid.sum())
    if denom == 0:
        return 0.0
    changed = (semantic_before != semantic_after) & valid
    return float(changed.sum()) / float(denom)


def transition_similarity(hist_a: np.ndarray, hist_b: np.ndarray, metric: str = "cosine") -> float:
    if metric != "cosine":
        raise ValueError(f"Unsupported metric={metric}")
    denom = float(np.linalg.norm(hist_a) * np.linalg.norm(hist_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(hist_a, hist_b) / denom)
