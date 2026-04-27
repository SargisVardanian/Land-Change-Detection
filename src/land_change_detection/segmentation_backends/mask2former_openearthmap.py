from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import SegmentationArtifact
from ..semantic_hf import DEFAULT_CLASS_COLORS, Mask2FormerSatelliteSegmenter
from .base import BaseSegmentationBackend


class Mask2FormerOpenEarthMapBackend(BaseSegmentationBackend):
    def __init__(self, model_dir: str | Path, device: str):
        self.segmenter = Mask2FormerSatelliteSegmenter(model_dir=model_dir, device=device)

    def predict(self, image: np.ndarray) -> SegmentationArtifact:
        result = self.segmenter.segment(image)
        return SegmentationArtifact(
            semantic_map=result.class_map,
            confidence_map=None,
            label_summary=result.label_summary,
            overlay_rgb=result.color_map,
            legend=self.segmenter.id2label,
            class_colors=DEFAULT_CLASS_COLORS.copy(),
            metadata={"backend": "mask2former_openearthmap", "confidence_available": False},
        )
