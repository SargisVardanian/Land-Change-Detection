from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceClassDef:
    id: int
    name: str
    color: str
    label_ru: str


# WorldCover-aligned base schema for the semantic-first pipeline.
SURFACE_CLASS_DEFS: tuple[SurfaceClassDef, ...] = (
    SurfaceClassDef(0, "unknown", "#6c757d", "Неизвестно"),
    SurfaceClassDef(1, "tree_cover", "#006400", "Древесный покров"),
    SurfaceClassDef(2, "shrubland", "#ffbb22", "Кустарники"),
    SurfaceClassDef(3, "grassland", "#ffff4c", "Травянистая растительность"),
    SurfaceClassDef(4, "cropland", "#f096ff", "Сельхозугодья"),
    SurfaceClassDef(5, "built_up", "#fa0000", "Застройка"),
    SurfaceClassDef(6, "bare_sparse", "#b4b4b4", "Голая / разреженная поверхность"),
    SurfaceClassDef(7, "snow_ice", "#f0f0f0", "Снег / лёд"),
    SurfaceClassDef(8, "water", "#0064c8", "Вода"),
    SurfaceClassDef(9, "wetland", "#0096a0", "Водно-болотные угодья"),
    SurfaceClassDef(10, "mangroves", "#00cf75", "Мангровые леса"),
    SurfaceClassDef(11, "moss_lichen", "#fae6a0", "Мхи / лишайники"),
)

SURFACE_NAME_TO_ID = {item.name: item.id for item in SURFACE_CLASS_DEFS}
SURFACE_ID_TO_NAME = {item.id: item.name for item in SURFACE_CLASS_DEFS}

# Prithvi-EO-2.0 expects six EO channels in this order.
PRITHVI_BAND_NAMES: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")

# OSCD 13-band stack indices mapped to Prithvi-style 6-band inputs.
# B02 blue, B03 green, B04 red, B8A narrow NIR, B11 SWIR1, B12 SWIR2.
PRITHVI_OSCD_INDICES: tuple[int, ...] = (1, 2, 3, 8, 11, 12)


def select_prithvi_bands(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected (C,H,W) array, got {arr.shape}")
    if arr.shape[0] < max(PRITHVI_OSCD_INDICES) + 1:
        raise ValueError(f"expected at least 13 bands, got shape={arr.shape}")
    return arr[list(PRITHVI_OSCD_INDICES)].astype(np.float32, copy=False)


def transition_label(before_id: int, after_id: int) -> str:
    before_name = SURFACE_ID_TO_NAME.get(int(before_id), "unknown")
    after_name = SURFACE_ID_TO_NAME.get(int(after_id), "unknown")
    if before_name == after_name:
        return "no_change"
    return f"{before_name}->{after_name}"


def diff_surface_maps(before_map: np.ndarray, after_map: np.ndarray) -> np.ndarray:
    if before_map.shape != after_map.shape:
        raise ValueError(f"surface maps must match, got {before_map.shape} vs {after_map.shape}")
    return (before_map != after_map).astype(np.uint8)


def summarize_surface_classes(class_map: np.ndarray) -> list[dict]:
    total = max(int(class_map.size), 1)
    summary: list[dict] = []
    for item in SURFACE_CLASS_DEFS:
        pixels = int((class_map == item.id).sum())
        summary.append(
            {
                "id": item.id,
                "name": item.name,
                "label_ru": item.label_ru,
                "color": item.color,
                "pixels": pixels,
                "percent": round(pixels * 100.0 / total, 3),
            }
        )
    return summary


def summarize_transitions(
    before_map: np.ndarray,
    after_map: np.ndarray,
    top_k: int = 10,
    id2label: dict[int, str] | None = None,
) -> list[dict]:
    if before_map.shape != after_map.shape:
        raise ValueError(f"surface maps must match, got {before_map.shape} vs {after_map.shape}")

    valid = (before_map >= 0) & (after_map >= 0)
    if not np.any(valid):
        return []

    before_flat = before_map[valid].astype(np.int32, copy=False)
    after_flat = after_map[valid].astype(np.int32, copy=False)
    encoded = before_flat * 100 + after_flat
    uniq, counts = np.unique(encoded, return_counts=True)
    order = np.argsort(counts)[::-1]

    result: list[dict] = []
    total = int(valid.sum())
    label_map = {int(key): str(value) for key, value in (id2label or {}).items()} or SURFACE_ID_TO_NAME
    for token, count in zip(uniq[order][:top_k], counts[order][:top_k], strict=False):
        before_id = int(token // 100)
        after_id = int(token % 100)
        before_name = label_map.get(before_id, f"class_{before_id}")
        after_name = label_map.get(after_id, f"class_{after_id}")
        result.append(
            {
                "before_id": before_id,
                "after_id": after_id,
                "before": before_name,
                "after": after_name,
                "transition": "no_change" if before_name == after_name else f"{before_name}->{after_name}",
                "pixels": int(count),
                "percent": round(int(count) * 100.0 / total, 3),
            }
        )
    return result
