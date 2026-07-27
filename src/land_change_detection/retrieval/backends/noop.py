from __future__ import annotations

from ..contracts import RetrievalArtifact, RetrievalQuery, RetrievalResult
from .base import RetrievalBackend


class NoOpRetrievalBackend(RetrievalBackend):
    def retrieve(self, query: RetrievalQuery) -> RetrievalArtifact:
        return RetrievalArtifact(
            mode=query.mode,
            result=RetrievalResult(
                query=query,
                items=[],
                backend_name="noop",
                metadata={"deterministic": True, "reason": "no-op backend"},
            ),
            metadata={"item_count": 0},
        )
