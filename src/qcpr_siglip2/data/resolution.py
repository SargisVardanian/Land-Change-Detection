"""Auditable effective-resolution records for direct and hierarchical NaFlex."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Any, Literal

from .chunking import SynchronizedChunkPlan, chunk_plan_coverage_fraction


def use_partition_mode(
    item_id: str, *, epoch: int, fraction: float, seed: int
) -> bool:
    """Select a bounded deterministic subset for hierarchy augmentation."""

    if not item_id:
        raise ValueError("item_id must be non-empty")
    if epoch < 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("epoch and fraction are outside the valid range")
    payload = f"{seed}:{epoch}:{item_id}".encode()
    value = int.from_bytes(sha256(payload).digest()[:8], "big") / 2**64
    return value < fraction


def effective_resolution_record(
    *,
    native_height: int,
    native_width: int,
    patch_size: int,
    actual_valid_patch_count: int,
    max_num_patches: int,
    processing_mode: Literal["DIRECT_NAFLEX", "HIERARCHICAL_NATIVE"],
    processed_patch_grid: tuple[int, int] | None = None,
    chunk_plan: SynchronizedChunkPlan | None = None,
    overview_patch_grid: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Describe how much native spatial information reached the visual tower."""

    if min(native_height, native_width, patch_size, max_num_patches) <= 0:
        raise ValueError("resolution dimensions, patch size and budget must be positive")
    native_patch_count = math.ceil(native_height / patch_size) * math.ceil(
        native_width / patch_size
    )
    if processing_mode == "DIRECT_NAFLEX":
        if processed_patch_grid is None or chunk_plan is not None:
            raise ValueError("direct mode requires one processed patch grid")
        grid_height, grid_width = processed_patch_grid
        processed_height = min(native_height, grid_height * patch_size)
        processed_width = min(native_width, grid_width * patch_size)
        preserved = min(1.0, actual_valid_patch_count / native_patch_count)
        coverage = preserved
        tiles = 1
    else:
        if chunk_plan is None:
            raise ValueError("hierarchical mode requires a synchronized chunk plan")
        processed_height = native_height
        processed_width = native_width
        coverage = chunk_plan_coverage_fraction(chunk_plan)
        preserved = coverage
        tiles = len(chunk_plan.chunks)
    return {
        "native_width": native_width,
        "native_height": native_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "patch_size": patch_size,
        "native_patch_count_if_full_resolution": native_patch_count,
        "actual_valid_patch_count": actual_valid_patch_count,
        "max_num_patches": max_num_patches,
        "scale_x": processed_width / native_width,
        "scale_y": processed_height / native_height,
        "native_information_preserved_fraction": preserved,
        "native_pixel_coverage_fraction": coverage,
        "processing_mode": processing_mode,
        "tile_count": tiles,
        "overview_processed_patch_grid": (
            list(overview_patch_grid) if overview_patch_grid is not None else None
        ),
    }
