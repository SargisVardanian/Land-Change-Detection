from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import SegmentationArtifact
from ..semantic_surface import PRITHVI_BAND_NAMES, select_prithvi_bands
from .base import BaseSegmentationBackend


class PrithviTerratorchBackend(BaseSegmentationBackend):
    """Planned multispectral Prithvi backend.

    Prithvi-EO-2.0 is not a drop-in RGB replacement for the current
    Mask2Former path. It expects EO bands and a TerraTorch task head, so this
    backend is intentionally explicit until training/inference config is added.
    """

    def __init__(self, model_dir: str | Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device

    def predict(self, image: np.ndarray) -> SegmentationArtifact:
        if image.ndim != 3 or image.shape[0] < 13 or image.shape[-1] in {3, 4}:
            raise ValueError(
                "expected Prithvi input as channel-first EO stack with at least 13 bands; "
                f"got shape={image.shape}"
            )
        bands = select_prithvi_bands(image)
        raise NotImplementedError(
            "prithvi_terratorch is a planned multispectral backend. "
            f"Received compatible bands {PRITHVI_BAND_NAMES} with shape {bands.shape}, "
            "but TerraTorch model/head wiring is not implemented yet."
        )
