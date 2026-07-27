from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts import RetrievalArtifact, RetrievalQuery


class RetrievalBackend(ABC):
    def __init__(self, model_dir: str | Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalArtifact:
        raise NotImplementedError
