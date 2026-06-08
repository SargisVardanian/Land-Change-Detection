from __future__ import annotations

from pathlib import Path

from .base import BaseSegmentationBackend


def get_segmentation_backend(backend_name: str, model_dir: str | Path, device: str) -> BaseSegmentationBackend:
    if backend_name == "mask2former_openearthmap":
        from .mask2former_openearthmap import Mask2FormerOpenEarthMapBackend

        return Mask2FormerOpenEarthMapBackend(model_dir=model_dir, device=device)
    if backend_name == "prithvi_terratorch":
        from .prithvi_terratorch import PrithviTerratorchBackend

        return PrithviTerratorchBackend(model_dir=model_dir, device=device)
    raise ValueError(f"Unknown segmentation backend: {backend_name}")
