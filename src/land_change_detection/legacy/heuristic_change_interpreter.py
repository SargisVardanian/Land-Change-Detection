from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def focus_square_from_pixel_delta(before_rgb: np.ndarray, after_rgb: np.ndarray) -> str:
    before_arr = np.asarray(before_rgb, dtype=np.float32)
    after_arr = np.asarray(after_rgb, dtype=np.float32)
    delta = np.abs(before_arr - after_arr).mean(axis=2)
    height, width = delta.shape
    windows = {
        "top-left": delta[: height // 2, : width // 2],
        "top-right": delta[: height // 2, width // 2 :],
        "center": delta[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4],
        "bottom-left": delta[height // 2 :, : width // 2],
        "bottom-right": delta[height // 2 :, width // 2 :],
    }
    best = max(windows.items(), key=lambda item: float(item[1].mean()) if item[1].size else -1.0)
    return best[0]


def mean_pixel_delta(before_rgb: np.ndarray, after_rgb: np.ndarray) -> float:
    before_arr = np.asarray(before_rgb, dtype=np.float32)
    after_arr = np.asarray(after_rgb, dtype=np.float32)
    return float(np.abs(before_arr - after_arr).mean())


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(np.clip(rgb, 0, 255), dtype=np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]


def edge_strength_map(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return dx + dy


@dataclass(frozen=True)
class ChangeCellEvidence:
    cell_id: str
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    mean_delta: float
    edge_before: float
    edge_after: float
    edge_delta: float
    brightness_before: float
    brightness_after: float
    brightness_delta: float
    bright_patch_before: float
    bright_patch_after: float
    bright_patch_delta: float
    dark_patch_before: float
    dark_patch_after: float
    dark_patch_delta: float
    bright_roof_before: float
    bright_roof_after: float
    green_before: float
    green_after: float
    score: float
    descriptors: list[str]


@dataclass(frozen=True)
class ChangeEvidence:
    grid_size: int
    cells: list[ChangeCellEvidence]
    top_cells: list[ChangeCellEvidence]
    global_summary: str


def grid_cell_name(row_idx: int, col_idx: int, grid_size: int) -> str:
    row_label = chr(ord("A") + row_idx)
    return f"{row_label}{col_idx + 1}"


def green_pixel_ratio(tile: np.ndarray) -> float:
    arr = np.asarray(tile, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if arr.max() <= 1.5:
        arr = arr * 255.0
    greenish = (arr[..., 1] > arr[..., 0] * 1.08) & (arr[..., 1] > arr[..., 2] * 1.05) & (arr[..., 1] > 70)
    return float(greenish.mean())


def bright_roof_ratio(tile: np.ndarray, gray_tile: np.ndarray, edge_tile: np.ndarray) -> float:
    arr = np.asarray(tile, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if arr.max() <= 1.5:
        arr = arr * 255.0
    channel_spread = arr.max(axis=2) - arr.min(axis=2)
    roof_like = (gray_tile > 145) & (gray_tile < 235) & (channel_spread < 70) & (edge_tile > 18)
    return float(roof_like.mean())


def interpret_cell_change(cell: ChangeCellEvidence) -> list[str]:
    interpretations: list[str] = []
    rect_bright_delta = cell.bright_roof_after - cell.bright_roof_before
    strong_rectilinear = rect_bright_delta >= 0.075 and cell.edge_delta >= 8 and cell.score >= 55
    medium_rectilinear = rect_bright_delta >= 0.045 and cell.edge_delta >= 5 and cell.score >= 35
    if strong_rectilinear:
        interpretations.append("likely rectilinear surface feature was added or expanded in AFTER")
    elif medium_rectilinear:
        interpretations.append("possible rectilinear surface feature appears in AFTER")
    if cell.dark_patch_delta >= 0.065 and cell.edge_delta >= 7 and cell.score >= 45:
        interpretations.append("likely compact hard-surface or disturbed yard expands in AFTER")
    elif cell.dark_patch_delta >= 0.035:
        interpretations.append("possible compacted or disturbed ground expands in AFTER")
    if cell.edge_delta >= 10 and (cell.bright_patch_delta >= 0.035 or rect_bright_delta >= 0.025):
        interpretations.append("likely access-line, boundary, or subdivision trace appears in AFTER")
    elif cell.edge_delta >= 7:
        interpretations.append("possible access-line or service-edge rework appears in AFTER")
    if cell.bright_patch_delta >= 0.08 and cell.brightness_delta >= 8:
        interpretations.append("likely fresh grading or exposed prepared ground expands in AFTER")
    elif cell.bright_patch_delta >= 0.045:
        interpretations.append("possible bare-ground expansion or graded platform appears in AFTER")
    if cell.edge_delta <= -8 and cell.brightness_delta >= 6:
        interpretations.append("a rougher surface is replaced by smoother cleared ground in AFTER")
    if not interpretations and cell.mean_delta >= 12:
        interpretations.append("visible surface brightness/texture changed, but no object-level conversion is clear")
    if not interpretations:
        interpretations.append("stable area / no clear object-level change detected")
    return interpretations[:3]


def summarize_cell_support(cell: ChangeCellEvidence) -> str:
    signals = []
    if cell.edge_delta >= 8:
        signals.append("higher linear density")
    elif cell.edge_delta <= -8:
        signals.append("lower linear density")
    if cell.brightness_delta >= 8:
        signals.append("brighter surface")
    elif cell.brightness_delta <= -8:
        signals.append("darker surface")
    if cell.bright_patch_delta >= 0.05:
        signals.append("more exposed bright ground")
    if cell.dark_patch_delta >= 0.03:
        signals.append("more dark compact patches")
    rect_bright_delta = cell.bright_roof_after - cell.bright_roof_before
    if rect_bright_delta >= 0.035:
        signals.append("more rectilinear bright features")
    if not signals:
        signals.append("weak local difference")
    return ", ".join(signals[:3])


def describe_cell_objects(cell: ChangeCellEvidence, phase: str) -> str:
    if phase == "before":
        edge = cell.edge_before
        bright_patch = cell.bright_patch_before
        dark_patch = cell.dark_patch_before
        bright_roof = cell.bright_roof_before
        green = cell.green_before
    else:
        edge = cell.edge_after
        bright_patch = cell.bright_patch_after
        dark_patch = cell.dark_patch_after
        bright_roof = cell.bright_roof_after
        green = cell.green_after
    objects: list[str] = []
    if bright_roof >= 0.06:
        objects.append("rectilinear bright hard-surface patches")
    elif bright_roof >= 0.025:
        objects.append("small rectilinear bright traces")
    if dark_patch >= 0.10:
        objects.append("dark compact or paved surfaces")
    elif dark_patch >= 0.04:
        objects.append("small dark compact features")
    if edge >= 34:
        objects.append("dense linear road/plot texture")
    elif edge >= 23:
        objects.append("visible road or plot edges")
    if bright_patch >= 0.30:
        objects.append("large bright exposed soil or graded ground")
    elif bright_patch >= 0.12:
        objects.append("bright bare-soil patches")
    if green >= 0.05:
        objects.append("sparse vegetation patches")
    if not objects:
        objects.append("mostly bare arid ground")
    return "; ".join(objects[:4])


def cell_confidence(cell: ChangeCellEvidence) -> str:
    if cell.score >= 65 or cell.mean_delta >= 18:
        return "high"
    if cell.score >= 35 or cell.mean_delta >= 8:
        return "medium"
    return "low"


def physical_interpretation_for_cell(cell: ChangeCellEvidence) -> str:
    likely = interpret_cell_change(cell)
    confidence = cell_confidence(cell)
    prefix = "Likely" if confidence == "high" else "Possible" if confidence == "medium" else "Weakly supported"
    if any("rectilinear surface feature" in item for item in likely):
        return f"{prefix} rectilinear surface-feature change. It may reflect a hardstand, prepared ground, slab-like surface, or another compact feature; object identity remains cautious."
    if any("compacted" in item or "hard-surface" in item or "disturbed ground" in item for item in likely):
        return f"{prefix} compacted or disturbed-surface change; exact object type remains uncertain at this resolution."
    if any("boundary" in item or "subdivision" in item or "access-line" in item or "service-edge" in item for item in likely):
        return f"{prefix} access-line, boundary, service-edge, or subdivision trace."
    if any("grading" in item or "bare-ground" in item or "graded platform" in item or "clearing" in item for item in likely):
        return f"{prefix} exposed prepared ground or grading; this suggests surface work, scraping, fill, or clearing."
    if any("stable area" in item or "no clear" in item for item in likely):
        return "No stable object-level conversion is visible beyond weak local differences or registration noise."
    return "Visible surface brightness or texture changed, but no clear road, building, excavation, or field conversion can be identified."


def build_cell_report_rows(evidence: ChangeEvidence) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in sorted(evidence.cells, key=lambda item: item.cell_id):
        rows.append(
            {
                "cell": cell.cell_id,
                "objects_before": describe_cell_objects(cell, "before"),
                "objects_after": describe_cell_objects(cell, "after"),
                "likely_change": "; ".join(interpret_cell_change(cell)),
                "technical_interpretation": physical_interpretation_for_cell(cell),
                "support": summarize_cell_support(cell),
                "score": round(cell.score, 1),
                "confidence": cell_confidence(cell),
            }
        )
    return rows


def build_scene_overview(evidence: ChangeEvidence) -> str:
    top_cells = evidence.top_cells
    top_cell_ids = ", ".join(cell.cell_id for cell in top_cells) or "none"
    all_cells = evidence.cells
    cell_interpretations = {cell.cell_id: interpret_cell_change(cell) for cell in all_cells}
    category_counts = {
        "grading/clearing-like surface preparation": 0,
        "compact-dark-surface or yard-like change": 0,
        "rectilinear surface feature change": 0,
        "road/access/line-like rework": 0,
        "ambiguous local reworking": 0,
        "stable/no clear object-level change": 0,
    }
    for interpretations in cell_interpretations.values():
        joined = " ".join(interpretations)
        if "grading" in joined or "bare-ground" in joined or "smoother cleared" in joined:
            category_counts["grading/clearing-like surface preparation"] += 1
        if "compacted" in joined or "paved-service" in joined or "disturbed ground" in joined:
            category_counts["compact-dark-surface or yard-like change"] += 1
        if "rectilinear surface feature" in joined:
            category_counts["rectilinear surface feature change"] += 1
        if "access-line" in joined or "service-edge" in joined or "boundary" in joined or "subdivision" in joined:
            category_counts["road/access/line-like rework"] += 1
        if "ambiguous" in joined:
            category_counts["ambiguous local reworking"] += 1
        if "stable area" in joined:
            category_counts["stable/no clear object-level change"] += 1
    strong_cells = [cell for cell in evidence.cells if cell.score >= 45]
    if len(strong_cells) <= 4:
        locality = "concentrated in a few cells"
    elif len(strong_cells) <= 8:
        locality = "clustered across several neighboring cells"
    else:
        locality = "distributed across much of the crop"
    dominant = [
        f"{label} ({count}/16 cells)"
        for label, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
        if count > 0
    ][:3]
    if not dominant:
        dominant = ["weak local differences without a stable object-level conversion"]
    return (
        f"Change is {locality}, mainly around {top_cell_ids}. "
        "The dominant patterns are "
        + "; ".join(dominant)
        + ". These interpretations are conservative: rectilinear bright cues are treated as surface features, prepared ground, or compact surfaces unless the imagery gives stronger object-level evidence."
    )
