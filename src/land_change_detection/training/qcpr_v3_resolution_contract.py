from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CropGeometry:
    box: tuple[int, int, int, int]
    raw_foreground: int
    cropped_foreground: int
    resized_foreground: int
    canonical_foreground: int


def foreground_preserving_target(mask: Tensor, grid: int = 32) -> Tensor:
    """Downsample binary supervision without deleting small foreground."""
    if mask.ndim != 2 or grid <= 0:
        raise ValueError("mask must be [H,W] and grid must be positive")
    return F.adaptive_max_pool2d(mask.float()[None, None], (grid, grid))[0, 0]


def target_bbox(mask: Tensor) -> tuple[int, int, int, int] | None:
    points = torch.nonzero(mask > 0, as_tuple=False)
    if not points.numel():
        return None
    y0, x0 = points.amin(0).tolist(); y1, x1 = (points.amax(0) + 1).tolist()
    return int(y0), int(y1), int(x0), int(x1)


def target_aware_crop_box(
    mask: Tensor, *, output_size: int = 256, context: float = 1.5,
    jitter_seed: int | None = None,
) -> tuple[int, int, int, int]:
    """Square crop containing all target pixels, optionally translated deterministically.

    Jitter changes only the crop origin within the range that still contains the
    complete target. It prevents an absolute-coordinate shortcut without
    changing visual ground truth or crop scale.
    """
    if mask.ndim != 2 or output_size <= 0 or context < 1:
        raise ValueError("invalid target-aware crop arguments")
    height, width = mask.shape
    bbox = target_bbox(mask)
    if bbox is None:
        raise ValueError("target-aware positive crop requires a non-empty mask")
    y0, y1, x0, x1 = bbox
    side = min(max(int(round(max(y1 - y0, x1 - x0) * context)), output_size), height, width)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    centered_top = min(max(cy - side // 2, 0), height - side)
    centered_left = min(max(cx - side // 2, 0), width - side)
    if jitter_seed is None:
        top, left = centered_top, centered_left
    else:
        min_top, max_top = max(0, y1 - side), min(y0, height - side)
        min_left, max_left = max(0, x1 - side), min(x0, width - side)
        generator = torch.Generator().manual_seed(int(jitter_seed))
        top = int(torch.randint(min_top, max_top + 1, (), generator=generator)) if max_top > min_top else min_top
        left = int(torch.randint(min_left, max_left + 1, (), generator=generator)) if max_left > min_left else min_left
    return top, top + side, left, left + side


def apply_spatial_contract(t1: Tensor, t2: Tensor, mask: Tensor, box: tuple[int, int, int, int], *, output_size: int = 256) -> tuple[Tensor, Tensor, Tensor]:
    if t1.shape != t2.shape or t1.ndim != 3 or mask.shape != t1.shape[-2:]:
        raise ValueError("T1, T2 and mask geometry must align")
    y0, y1, x0, x1 = box
    first = F.interpolate(t1[:, y0:y1, x0:x1][None], (output_size, output_size), mode="bilinear", align_corners=False)[0]
    second = F.interpolate(t2[:, y0:y1, x0:x1][None], (output_size, output_size), mode="bilinear", align_corners=False)[0]
    target = F.interpolate(mask[y0:y1, x0:x1][None, None].float(), (output_size, output_size), mode="nearest")[0, 0]
    return first, second, target


def reverse_directional_sample(t1: Tensor, t2: Tensor, appeared: Tensor, disappeared: Tensor, query: str) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
    lower = query.casefold()
    if "appeared" in lower or "new" in lower or "built" in lower:
        swapped = "buildings disappeared"
    elif "disappeared" in lower or "demolished" in lower or "removed" in lower:
        swapped = "new buildings appeared"
    else:
        raise ValueError("query has no audited temporal direction")
    return t2, t1, disappeared, appeared, swapped
