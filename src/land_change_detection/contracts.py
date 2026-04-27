from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentationArtifact:
    semantic_map: np.ndarray
    confidence_map: np.ndarray | None
    label_summary: list[dict[str, Any]]
    overlay_rgb: np.ndarray
    legend: dict[int, str]
    class_colors: dict[str, str] = field(default_factory=dict)
    features: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def class_map(self) -> np.ndarray:
        return self.semantic_map

    @property
    def color_map(self) -> np.ndarray:
        return self.overlay_rgb

    @property
    def id2label(self) -> dict[int, str]:
        return self.legend


@dataclass(frozen=True)
class CellRegion:
    cell_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class CellPack:
    cell: CellRegion
    before_crop: np.ndarray
    after_crop: np.ndarray
    before_semantic_map: np.ndarray
    after_semantic_map: np.ndarray
    before_overlay: np.ndarray
    after_overlay: np.ndarray
    before_confidence: np.ndarray | None
    after_confidence: np.ndarray | None
    legend: dict[int, str]
    scene_thumbnail: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineV2Result:
    before_segmentation: SegmentationArtifact
    after_segmentation: SegmentationArtifact
    cell_packs: list[CellPack]
    scene_report: Any | None = None
    metadata: dict[str, Any] | None = None
