from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


TimeSemantics = Literal["ordinal_not_calendar", "calendar_interval"]


@dataclass(frozen=True)
class TemporalContext:
    order: tuple[str, str] = ("before", "after")
    t1_timestamp: str | None = None
    t2_timestamp: str | None = None
    delta_days: float | None = None
    duration_known: bool = False
    time_semantics: TimeSemantics = "ordinal_not_calendar"

    def to_dict(self) -> dict:
        return {
            "order": list(self.order),
            "t1_timestamp": self.t1_timestamp,
            "t2_timestamp": self.t2_timestamp,
            "delta_days": self.delta_days,
            "duration_known": self.duration_known,
            "time_semantics": self.time_semantics,
        }


@dataclass(frozen=True)
class ComponentTargets:
    masks: Tensor
    full_resolution_mask: Tensor
    areas: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(self.masks.shape[0])


def binarize_mask(mask: Tensor, threshold: float = 0.5) -> Tensor:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H,W] or [1,H,W].")
    return mask.to(torch.float32) >= threshold


def connected_components_8(mask: Tensor, min_area: int = 4) -> list[Tensor]:
    binary = binarize_mask(mask)
    height, width = binary.shape
    visited = torch.zeros_like(binary, dtype=torch.bool)
    components: list[Tensor] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            comp = torch.zeros_like(binary, dtype=torch.bool)
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                comp[cy, cx] = True
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            if int(comp.sum().item()) >= min_area:
                components.append(comp)
    return components


def resize_component_masks(components: list[Tensor], grid_size: tuple[int, int] = (36, 36)) -> Tensor:
    if not components:
        return torch.zeros(0, grid_size[0] * grid_size[1], dtype=torch.float32)
    stacked = torch.stack([component.to(torch.float32) for component in components], dim=0).unsqueeze(1)
    resized = F.interpolate(stacked, size=grid_size, mode="nearest").squeeze(1)
    return resized.flatten(1)


def build_component_targets(mask: Tensor, grid_size: tuple[int, int] = (36, 36), min_area: int = 4) -> ComponentTargets:
    binary = binarize_mask(mask)
    components = connected_components_8(binary, min_area=min_area)
    areas = tuple(int(component.sum().item()) for component in components)
    return ComponentTargets(
        masks=resize_component_masks(components, grid_size=grid_size),
        full_resolution_mask=binary.to(torch.float32),
        areas=areas,
    )
