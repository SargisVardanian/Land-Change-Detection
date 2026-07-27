from __future__ import annotations

from pathlib import Path

from .contracts import RetrievalArtifact, RetrievalQuery
from .registry import get_retrieval_backend


class RetrievalRuntime:
    def __init__(self, backend_name: str, model_dir: str | Path = ".", device: str = "cpu"):
        self.backend_name = backend_name
        self.model_dir = Path(model_dir)
        self.device = device
        self.backend = get_retrieval_backend(
            backend_name=backend_name,
            model_dir=self.model_dir,
            device=device,
        )

    def run(self, query: RetrievalQuery) -> RetrievalArtifact:
        return self.backend.retrieve(query)
