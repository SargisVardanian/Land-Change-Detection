from __future__ import annotations

import numpy as np


def confidence_to_heatmap(confidence_map: np.ndarray | None) -> np.ndarray | None:
    if confidence_map is None:
        return None
    arr = np.asarray(confidence_map, dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.clip(arr, 0.0, 1.0)
    heat = np.zeros((*arr.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip((1.0 - arr) * 255.0, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    heat[..., 2] = 48
    return heat


def label_legend_rows(legend: dict[int, str], class_colors: dict[str, str]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for class_id, label in sorted(legend.items()):
        rows.append(
            {
                "id": class_id,
                "label": label,
                "color": class_colors.get(label, "#95a5a6"),
            }
        )
    return rows
