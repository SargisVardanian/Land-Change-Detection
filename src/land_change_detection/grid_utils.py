from __future__ import annotations

import numpy as np

from .contracts import CellPack, CellRegion, SegmentationArtifact


def build_grid_regions(width: int, height: int, rows: int = 4, cols: int = 4) -> list[CellRegion]:
    row_bounds = np.linspace(0, height, rows + 1, dtype=int)
    col_bounds = np.linspace(0, width, cols + 1, dtype=int)
    regions: list[CellRegion] = []
    for row_idx in range(rows):
        row_label = chr(ord("A") + row_idx)
        for col_idx in range(cols):
            x0 = int(col_bounds[col_idx])
            x1 = int(col_bounds[col_idx + 1])
            y0 = int(row_bounds[row_idx])
            y1 = int(row_bounds[row_idx + 1])
            regions.append(
                CellRegion(
                    cell_id=f"{row_label}{col_idx + 1}",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )
    return regions


def build_cell_packs(
    before_img: np.ndarray,
    after_img: np.ndarray,
    before_seg: SegmentationArtifact,
    after_seg: SegmentationArtifact,
    legend: dict[int, str],
    rows: int = 4,
    cols: int = 4,
    scene_thumbnail: np.ndarray | None = None,
) -> list[CellPack]:
    height, width = before_img.shape[:2]
    packs: list[CellPack] = []
    for region in build_grid_regions(width=width, height=height, rows=rows, cols=cols):
        packs.append(
            CellPack(
                cell=region,
                before_crop=before_img[region.y0 : region.y1, region.x0 : region.x1],
                after_crop=after_img[region.y0 : region.y1, region.x0 : region.x1],
                before_semantic_map=before_seg.semantic_map[region.y0 : region.y1, region.x0 : region.x1],
                after_semantic_map=after_seg.semantic_map[region.y0 : region.y1, region.x0 : region.x1],
                before_overlay=before_seg.overlay_rgb[region.y0 : region.y1, region.x0 : region.x1],
                after_overlay=after_seg.overlay_rgb[region.y0 : region.y1, region.x0 : region.x1],
                before_confidence=None if before_seg.confidence_map is None else before_seg.confidence_map[region.y0 : region.y1, region.x0 : region.x1],
                after_confidence=None if after_seg.confidence_map is None else after_seg.confidence_map[region.y0 : region.y1, region.x0 : region.x1],
                legend=legend,
                scene_thumbnail=scene_thumbnail,
            )
        )
    return packs
