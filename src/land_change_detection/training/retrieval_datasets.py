from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRecord:
    item_id: str
    mode: str
    features: list[float]
    text: str | None = None
    transition_label: str | None = None
    positives: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestRecord":
        return cls(
            item_id=str(payload["item_id"]),
            mode=str(payload["mode"]),
            features=[float(value) for value in payload.get("features", [])],
            text=str(payload["text"]) if payload.get("text") is not None else None,
            transition_label=str(payload["transition_label"]) if payload.get("transition_label") is not None else None,
            positives=tuple(str(value) for value in payload.get("positives", [])),
            negatives=tuple(str(value) for value in payload.get("negatives", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "mode": self.mode,
            "features": list(self.features),
            "text": self.text,
            "transition_label": self.transition_label,
            "positives": list(self.positives),
            "negatives": list(self.negatives),
            "metadata": dict(self.metadata),
        }


class ManifestRetrievalDataset:
    def __init__(self, records: list[ManifestRecord]):
        self.records = records
        self.by_id = {record.item_id: record for record in records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]

    def labels(self) -> list[str]:
        return [record.transition_label or record.item_id for record in self.records]

    def modes(self) -> list[str]:
        return [record.mode for record in self.records]


def load_manifest_records(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records: list[ManifestRecord] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(ManifestRecord.from_dict(json.loads(line)))
    return records
