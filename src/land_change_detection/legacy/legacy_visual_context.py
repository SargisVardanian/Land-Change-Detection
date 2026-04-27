from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .heuristic_change_interpreter import (
    ChangeCellEvidence,
    ChangeEvidence,
    bright_roof_ratio,
    build_scene_overview,
    edge_strength_map,
    focus_square_from_pixel_delta,
    green_pixel_ratio,
    grid_cell_name,
    interpret_cell_change,
    mean_pixel_delta,
    rgb_to_gray,
    summarize_cell_support,
)


def analyze_change_cells(before_rgb: np.ndarray, after_rgb: np.ndarray, grid_size: int = 4) -> ChangeEvidence:
    before_arr = np.asarray(before_rgb, dtype=np.float32)
    after_arr = np.asarray(after_rgb, dtype=np.float32)
    if before_arr.max() <= 1.5:
        before_arr = before_arr * 255.0
    if after_arr.max() <= 1.5:
        after_arr = after_arr * 255.0
    before_gray = rgb_to_gray(before_arr)
    after_gray = rgb_to_gray(after_arr)
    before_edges = edge_strength_map(before_gray)
    after_edges = edge_strength_map(after_gray)

    height, width = before_gray.shape
    row_bounds = np.linspace(0, height, grid_size + 1, dtype=int)
    col_bounds = np.linspace(0, width, grid_size + 1, dtype=int)
    results: list[ChangeCellEvidence] = []

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            top = int(row_bounds[row_idx])
            bottom = int(row_bounds[row_idx + 1])
            left = int(col_bounds[col_idx])
            right = int(col_bounds[col_idx + 1])
            before_tile = before_arr[top:bottom, left:right]
            after_tile = after_arr[top:bottom, left:right]
            before_gray_tile = before_gray[top:bottom, left:right]
            after_gray_tile = after_gray[top:bottom, left:right]
            before_edge_tile = before_edges[top:bottom, left:right]
            after_edge_tile = after_edges[top:bottom, left:right]
            if before_tile.size == 0 or after_tile.size == 0:
                continue

            mean_delta = float(np.abs(after_tile - before_tile).mean())
            edge_before = float(before_edge_tile.mean())
            edge_after = float(after_edge_tile.mean())
            edge_delta = edge_after - edge_before
            brightness_before = float(before_gray_tile.mean())
            brightness_after = float(after_gray_tile.mean())
            brightness_delta = brightness_after - brightness_before
            bright_patch_before = float((before_gray_tile > 190).mean())
            bright_patch_after = float((after_gray_tile > 190).mean())
            bright_patch_delta = bright_patch_after - bright_patch_before
            dark_patch_before = float((before_gray_tile < 85).mean())
            dark_patch_after = float((after_gray_tile < 85).mean())
            dark_patch_delta = dark_patch_after - dark_patch_before
            bright_roof_before = bright_roof_ratio(before_tile, before_gray_tile, before_edge_tile)
            bright_roof_after = bright_roof_ratio(after_tile, after_gray_tile, after_edge_tile)
            green_before_cell = green_pixel_ratio(before_tile)
            green_after_cell = green_pixel_ratio(after_tile)
            score = mean_delta + max(edge_delta, 0.0) * 1.8 + abs(brightness_delta) * 0.7

            descriptors: list[str] = []
            if mean_delta >= 10:
                descriptors.append("strong surface tone change")
            elif mean_delta >= 6:
                descriptors.append("moderate surface tone change")
            if edge_delta >= 8:
                descriptors.append("denser linear texture in AFTER")
            elif edge_delta <= -8:
                descriptors.append("simpler texture in AFTER")
            if brightness_delta >= 8:
                descriptors.append("brighter ground in AFTER")
            elif brightness_delta <= -8:
                descriptors.append("darker ground in AFTER")
            if bright_patch_delta >= 0.05:
                descriptors.append("new bright cleared or graded patches")
            if dark_patch_delta >= 0.03:
                descriptors.append("more dark compact features")
            if edge_delta >= 8 and bright_patch_delta >= 0.03:
                descriptors.append("new rectilinear traces likely visible")
            if not descriptors:
                descriptors.append("small but visible local change")

            results.append(
                ChangeCellEvidence(
                    cell_id=grid_cell_name(row_idx, col_idx, grid_size),
                    row=row_idx,
                    col=col_idx,
                    bbox=(left, top, right, bottom),
                    mean_delta=mean_delta,
                    edge_before=edge_before,
                    edge_after=edge_after,
                    edge_delta=edge_delta,
                    brightness_before=brightness_before,
                    brightness_after=brightness_after,
                    brightness_delta=brightness_delta,
                    bright_patch_before=bright_patch_before,
                    bright_patch_after=bright_patch_after,
                    bright_patch_delta=bright_patch_delta,
                    dark_patch_before=dark_patch_before,
                    dark_patch_after=dark_patch_after,
                    dark_patch_delta=dark_patch_delta,
                    bright_roof_before=bright_roof_before,
                    bright_roof_after=bright_roof_after,
                    green_before=green_before_cell,
                    green_after=green_after_cell,
                    score=score,
                    descriptors=descriptors[:3],
                )
            )
    sorted_cells = sorted(results, key=lambda item: float(item.score), reverse=True)
    global_summary = (
        f"4x4 grid over the crop. Strongest local changes cluster around "
        + ", ".join(cell.cell_id for cell in sorted_cells[:4])
        + ". Use cell IDs as stable spatial references."
    )
    return ChangeEvidence(grid_size=grid_size, cells=results, top_cells=sorted_cells[:4], global_summary=global_summary)


def build_change_guide_image(before_rgb: np.ndarray, after_rgb: np.ndarray, evidence: ChangeEvidence) -> np.ndarray:
    before_arr = np.asarray(np.clip(before_rgb, 0, 255), dtype=np.float32)
    after_arr = np.asarray(np.clip(after_rgb, 0, 255), dtype=np.float32)
    if before_arr.max() <= 1.5:
        before_arr = before_arr * 255.0
    if after_arr.max() <= 1.5:
        after_arr = after_arr * 255.0
    delta = np.abs(after_arr - before_arr).mean(axis=2)
    delta = delta / max(float(delta.max()), 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(delta * 255.0, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(delta * 180.0, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(40 + delta * 60.0, 0, 255).astype(np.uint8)
    blended = (0.45 * after_arr + 0.55 * heat).clip(0, 255).astype(np.uint8)
    guide = Image.fromarray(blended)
    draw = ImageDraw.Draw(guide)
    width, height = guide.size
    for row_idx in range(evidence.grid_size):
        y = int((row_idx / evidence.grid_size) * height)
        draw.line((0, y, width, y), fill="#ffffff", width=1)
    for col_idx in range(evidence.grid_size):
        x = int((col_idx / evidence.grid_size) * width)
        draw.line((x, 0, x, height), fill="#ffffff", width=1)
    colors = ["#ffffff", "#00d4ff", "#ffd400", "#ff7f50"]
    for idx, cell in enumerate(evidence.top_cells):
        left, top, right, bottom = cell.bbox
        draw.rectangle((left, top, right - 1, bottom - 1), outline=colors[idx % len(colors)], width=3)
    for cell in evidence.cells:
        left, top, right, bottom = cell.bbox
        draw.text((left + 4, top + 4), cell.cell_id, fill="#ffffff")
    return np.asarray(guide)


def _to_uint8_like(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def build_change_zoom_strip(before_rgb: np.ndarray, after_rgb: np.ndarray, evidence: ChangeEvidence, panel_size: int = 180) -> np.ndarray:
    panels: list[Image.Image] = []
    for cell in evidence.top_cells:
        left, top, right, bottom = cell.bbox
        before_crop = Image.fromarray(_to_uint8_like(before_rgb[top:bottom, left:right]))
        after_crop = Image.fromarray(_to_uint8_like(after_rgb[top:bottom, left:right]))
        before_panel = ImageOps.fit(before_crop, (panel_size, panel_size), method=Image.Resampling.BICUBIC)
        after_panel = ImageOps.fit(after_crop, (panel_size, panel_size), method=Image.Resampling.BICUBIC)
        pair = Image.new("RGB", (panel_size * 2 + 8, panel_size + 26), color=(0, 0, 0))
        pair_draw = ImageDraw.Draw(pair)
        pair.paste(before_panel, (0, 26))
        pair.paste(after_panel, (panel_size + 8, 26))
        pair_draw.text((4, 4), f"{cell.cell_id} before", fill="#ffffff")
        pair_draw.text((panel_size + 12, 4), f"{cell.cell_id} after", fill="#ffffff")
        panels.append(pair)
    if not panels:
        return np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    width = sum(img.size[0] for img in panels) + 12 * (len(panels) - 1)
    height = max(img.size[1] for img in panels)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0] + 12
    return np.asarray(canvas)


def build_visual_change_context(evidence: ChangeEvidence) -> str:
    category_overview = build_scene_overview(evidence)
    lines = [
        "Concise deterministic change-tool context for the same crop using a fixed 4x4 grid (A1..D4).",
        "Use BEFORE and AFTER as primary evidence. This text is only a compact guide.",
        "Do not automatically interpret bright rectilinear patches as buildings.",
        category_overview,
        evidence.global_summary,
        "Top changed cells ranked by change strength:",
    ]
    for idx, cell in enumerate(sorted(evidence.cells, key=lambda item: item.score, reverse=True)[:8], start=1):
        rect_bright_delta = cell.bright_roof_after - cell.bright_roof_before
        lines.append(
            "- "
            + f"{idx}. {cell.cell_id}: "
            + f"score {cell.score:.1f}; "
            + "candidates: "
            + "; ".join(interpret_cell_change(cell))
            + "; support: "
            + summarize_cell_support(cell)
            + f"; cues: edge={cell.edge_delta:.1f}, bright={cell.bright_patch_delta:.3f}, dark={cell.dark_patch_delta:.3f}, rect={rect_bright_delta:.3f}"
        )
    return "\n".join(lines)


def dominant_labels(label_summary: list[dict], min_percent: float = 5.0) -> list[str]:
    return [f"{item['label']} {item['percent']:.1f}%" for item in label_summary if item["percent"] >= min_percent]


def build_deterministic_breakdown(
    crop_before: np.ndarray,
    crop_after: np.ndarray,
    crop_before_result,
    crop_after_result,
    crop_transitions: list[dict],
) -> list[dict[str, str]]:
    evidence = analyze_change_cells(crop_before, crop_after, grid_size=4)
    rows = [
        {"field": "focus square by pixel delta", "value": focus_square_from_pixel_delta(crop_before, crop_after)},
        {"field": "mean pixel delta", "value": f"{mean_pixel_delta(crop_before, crop_after):.2f}"},
        {
            "field": "before dominant classes",
            "value": ", ".join(dominant_labels(crop_before_result.label_summary)) or "none above threshold",
        },
        {
            "field": "after dominant classes",
            "value": ", ".join(dominant_labels(crop_after_result.label_summary)) or "none above threshold",
        },
        {
            "field": "top semantic transitions",
            "value": ", ".join(f"{item['before']}->{item['after']} {item['percent']:.1f}%" for item in crop_transitions[:5]) or "no major transitions",
        },
    ]
    for cell in sorted(evidence.cells, key=lambda item: item.cell_id):
        rows.append(
            {
                "field": f"cell {cell.cell_id}",
                "value": f"likely change: {'; '.join(interpret_cell_change(cell))}; support: {summarize_cell_support(cell)}; score {cell.score:.1f}",
            }
        )
    return rows
