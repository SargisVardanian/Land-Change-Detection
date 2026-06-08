from __future__ import annotations

from ..contracts import RetrievalArtifact, RetrievalItem, RetrievalQuery, RetrievalResult
from .base import RetrievalBackend


class FakeRetrievalBackend(RetrievalBackend):
    def retrieve(self, query: RetrievalQuery) -> RetrievalArtifact:
        items: list[RetrievalItem] = []
        query_token = query.text or query.item_id or query.image_path or "query"
        for rank in range(1, query.top_k + 1):
            score = max(0.0, 1.0 - ((rank - 1) * 0.1))
            items.append(
                RetrievalItem(
                    item_id=f"{query.mode.value}:{query_token}:{rank}",
                    score=score,
                    mode=query.mode,
                    rank=rank,
                    metadata={
                        "query_token": query_token,
                        "backend": "fake",
                    },
                    thumbnail_path=f"/tmp/fake_{query.mode.value}_{rank}.png",
                    before_image_path=f"/tmp/fake_before_{rank}.png",
                    after_image_path=f"/tmp/fake_after_{rank}.png",
                    region={"x0": rank - 1, "y0": rank - 1, "x1": rank, "y1": rank},
                    transition_hint=query.filters.get("transition_hint"),
                )
            )
        result = RetrievalResult(
            query=query,
            items=items,
            backend_name="fake",
            metadata={"deterministic": True},
        )
        return RetrievalArtifact(
            mode=query.mode,
            result=result,
            metadata={"item_count": len(items)},
        )
