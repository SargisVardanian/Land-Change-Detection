from __future__ import annotations

import numpy as np

from .contracts import SegmentationArtifact
from .segmentation_backends.registry import get_segmentation_backend


class SegmentationRuntime:
    def __init__(self, backend_name: str, model_dir: str, device: str):
        self.backend_name = backend_name
        self.model_dir = model_dir
        self.device = device
        self.backend = get_segmentation_backend(backend_name=backend_name, model_dir=model_dir, device=device)

    def run(self, image: np.ndarray) -> SegmentationArtifact:
        return self.backend.predict(image)

    def legend(self) -> dict[int, str]:
        return getattr(self.backend, "segmenter").id2label
