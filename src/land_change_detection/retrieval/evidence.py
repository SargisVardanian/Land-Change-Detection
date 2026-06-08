from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import RetrievalArtifact


@dataclass(frozen=True)
class RetrievalEvidence:
    retrieval: RetrievalArtifact
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "summary": dict(self.summary),
        }


def build_retrieval_evidence(artifact: RetrievalArtifact) -> RetrievalEvidence:
    item_count = len(artifact.items)
    return RetrievalEvidence(
        retrieval=artifact,
        summary={
            "mode": artifact.mode.value,
            "backend_name": artifact.result.backend_name,
            "item_count": item_count,
            "top_item_id": artifact.items[0].item_id if item_count else None,
        },
    )
