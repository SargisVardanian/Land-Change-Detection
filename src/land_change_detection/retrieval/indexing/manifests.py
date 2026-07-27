from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import RetrievalMode


@dataclass(frozen=True)
class RetrievalManifestItem:
    item_id: str
    mode: RetrievalMode
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "mode": self.mode.value,
            "vector": list(self.vector),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalManifestItem":
        return cls(
            item_id=str(payload["item_id"]),
            mode=RetrievalMode(str(payload["mode"])),
            vector=[float(value) for value in payload["vector"]],
            metadata=dict(payload.get("metadata", {})),
        )
