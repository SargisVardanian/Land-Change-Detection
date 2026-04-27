from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..contracts import SegmentationArtifact


class BaseSegmentationBackend(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationArtifact:
        raise NotImplementedError
