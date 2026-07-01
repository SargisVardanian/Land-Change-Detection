from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TemporalChangeSample:
    images: Tensor
    timestamps: Tensor
    pair_id: str
    captions: list[str] = field(default_factory=list)
    temporal_valid_mask: Tensor | None = None
    binary_mask: Tensor | None = None
    semantic_masks: Tensor | None = None
    transition_labels: Tensor | None = None
    event_components: list[Any] | None = None
    event_texts: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.images.ndim != 4:
            raise ValueError(f"images must have shape [T,C,H,W], got {tuple(self.images.shape)}")
        if self.timestamps.ndim != 1 or self.timestamps.shape[0] != self.images.shape[0]:
            raise ValueError("timestamps must have shape [T] matching images")
        if self.temporal_valid_mask is not None and self.temporal_valid_mask.shape != self.timestamps.shape:
            raise ValueError("temporal_valid_mask must have shape [T]")


@dataclass(frozen=True)
class TemporalChangeBatch:
    images: Tensor
    timestamps: Tensor
    temporal_valid_mask: Tensor
    pair_ids: list[str]
    captions: list[list[str]]
    binary_mask: Tensor | None = None
    semantic_masks: Tensor | None = None
    transition_labels: Tensor | None = None
    event_components: list[Any] | None = None
    event_texts: list[list[str]] | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)
