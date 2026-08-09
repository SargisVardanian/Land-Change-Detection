"""Synchronized large-scene chunk plans for temporal imagery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise

import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class AlignedChunk:
    """One native-coordinate crop shared by every temporal frame."""

    chunk_index: int
    top: int
    left: int
    bottom: int
    right: int
    padded_height: int
    padded_width: int

    @property
    def valid_height(self) -> int:
        return self.bottom - self.top

    @property
    def valid_width(self) -> int:
        return self.right - self.left


@dataclass(frozen=True)
class SynchronizedChunkPlan:
    native_height: int
    native_width: int
    chunk_height: int
    chunk_width: int
    overlap_height: int
    overlap_width: int
    chunks: tuple[AlignedChunk, ...]
    transform_hash: str


def _axis_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= 0 or window <= 0:
        raise ValueError("axis length and chunk window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("chunk overlap must be in [0, window)")
    if length <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, length - window + 1, step))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def build_synchronized_chunk_plan(
    native_size: tuple[int, int],
    *,
    chunk_size: tuple[int, int],
    overlap: tuple[int, int] = (0, 0),
) -> SynchronizedChunkPlan:
    """Create a deterministic full-coverage crop plan in native coordinates."""

    native_height, native_width = map(int, native_size)
    chunk_height, chunk_width = map(int, chunk_size)
    overlap_height, overlap_width = map(int, overlap)
    rows = _axis_starts(native_height, chunk_height, overlap_height)
    columns = _axis_starts(native_width, chunk_width, overlap_width)
    chunks: list[AlignedChunk] = []
    for top in rows:
        for left in columns:
            chunks.append(
                AlignedChunk(
                    chunk_index=len(chunks),
                    top=top,
                    left=left,
                    bottom=min(top + chunk_height, native_height),
                    right=min(left + chunk_width, native_width),
                    padded_height=chunk_height,
                    padded_width=chunk_width,
                )
            )
    payload = {
        "native_size": [native_height, native_width],
        "chunk_size": [chunk_height, chunk_width],
        "overlap": [overlap_height, overlap_width],
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    transform_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SynchronizedChunkPlan(
        native_height=native_height,
        native_width=native_width,
        chunk_height=chunk_height,
        chunk_width=chunk_width,
        overlap_height=overlap_height,
        overlap_width=overlap_width,
        chunks=tuple(chunks),
        transform_hash=transform_hash,
    )


def apply_synchronized_chunk_plan(
    frames: list[Image.Image], plan: SynchronizedChunkPlan
) -> list[list[Image.Image]]:
    """Crop every frame identically and leave batch padding to NaFlex.

    Artificial pixel padding before the processor is forbidden because the
    vision transformer could attend to it as image content. The NaFlex
    processor owns padding and emits the authoritative patch-valid mask.
    """

    if len(frames) < 2:
        raise ValueError("large-scene temporal items require at least two frames")
    expected_size = (plan.native_width, plan.native_height)
    if any(frame.size != expected_size for frame in frames):
        raise ValueError("all temporal frames must match the chunk plan geometry")
    result: list[list[Image.Image]] = []
    for chunk in plan.chunks:
        temporal_chunk: list[Image.Image] = []
        for frame in frames:
            crop = frame.crop((chunk.left, chunk.top, chunk.right, chunk.bottom))
            temporal_chunk.append(crop)
        result.append(temporal_chunk)
    return result


def chunk_patch_coordinates(
    chunk: AlignedChunk,
    *,
    native_size: tuple[int, int],
    patch_grid: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return normalized native coordinates and valid centers for one chunk."""

    native_height, native_width = native_size
    grid_height, grid_width = patch_grid
    if min(native_height, native_width, grid_height, grid_width) <= 0:
        raise ValueError("native size and patch grid must be positive")
    ys = (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) / grid_height
    xs = (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) / grid_width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    local_y = yy * chunk.valid_height
    local_x = xx * chunk.valid_width
    valid = torch.ones_like(local_y, dtype=torch.bool)
    global_y = (chunk.top + local_y).clamp(max=float(native_height)) / native_height
    global_x = (chunk.left + local_x).clamp(max=float(native_width)) / native_width
    coordinates = torch.stack((global_x, global_y), dim=-1).reshape(-1, 2)
    return coordinates, valid.reshape(-1)


def chunk_plan_coverage_fraction(plan: SynchronizedChunkPlan) -> float:
    """Compute exact native-pixel coverage without allocating an image mask."""

    boundaries = sorted(
        {0, plan.native_height}
        | {chunk.top for chunk in plan.chunks}
        | {chunk.bottom for chunk in plan.chunks}
    )
    covered_area = 0
    for top, bottom in pairwise(boundaries):
        intervals = sorted(
            (chunk.left, chunk.right)
            for chunk in plan.chunks
            if chunk.top < bottom and chunk.bottom > top
        )
        merged_width = 0
        if intervals:
            left, right = intervals[0]
            for next_left, next_right in intervals[1:]:
                if next_left > right:
                    merged_width += right - left
                    left, right = next_left, next_right
                else:
                    right = max(right, next_right)
            merged_width += right - left
        covered_area += (bottom - top) * merged_width
    native_area = plan.native_height * plan.native_width
    return covered_area / native_area
