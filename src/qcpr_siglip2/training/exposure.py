"""Deterministic exposure accounting shared by bounded pilots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field


def sequence_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ExposureLedger:
    steps: int = 0
    pair_sequence: list[str] = field(default_factory=list)
    query_sequence: list[str] = field(default_factory=list)

    def record_step(self, pair_ids: Iterable[str], query_ids: Iterable[str]) -> None:
        self.steps += 1
        self.pair_sequence.extend(str(value) for value in pair_ids)
        self.query_sequence.extend(str(value) for value in query_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "physical_pair_presentations": len(self.pair_sequence),
            "query_presentations": len(self.query_sequence),
            "physical_pairs_unique": len(set(self.pair_sequence)),
            "query_unique": len(set(self.query_sequence)),
            "pair_sequence_sha256": sequence_sha256(self.pair_sequence),
            "query_sequence_sha256": sequence_sha256(self.query_sequence),
        }

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
