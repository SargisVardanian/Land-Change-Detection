"""Exposure and sequence accounting for controlled model pilots."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


def sha256_sequence(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


@dataclass
class ExposureAccounting:
    pair_sequence: list[str]
    query_sequence: list[str]
    per_step_pair_hashes: list[str]
    per_step_query_hashes: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_pair_presentations": len(self.pair_sequence),
            "total_query_presentations": len(self.query_sequence),
            "unique_pairs": len(set(self.pair_sequence)),
            "unique_queries": len(set(self.query_sequence)),
            "mean_presentations_per_pair": len(self.pair_sequence) / max(1, len(set(self.pair_sequence))),
            "mean_presentations_per_query": len(self.query_sequence) / max(1, len(set(self.query_sequence))),
            "pair_sequence_sha256": sha256_sequence(self.pair_sequence),
            "query_sequence_sha256": sha256_sequence(self.query_sequence),
            "per_step_pair_hashes": list(self.per_step_pair_hashes),
            "per_step_query_hashes": list(self.per_step_query_hashes),
            "pair_counts": dict(Counter(self.pair_sequence)),
            "query_counts": dict(Counter(self.query_sequence)),
        }


def record_step(
    accounting: ExposureAccounting,
    pair_ids: Sequence[str],
    query_ids: Sequence[str],
) -> None:
    accounting.pair_sequence.extend(str(item) for item in pair_ids)
    accounting.query_sequence.extend(str(item) for item in query_ids)
    accounting.per_step_pair_hashes.append(sha256_sequence(pair_ids))
    accounting.per_step_query_hashes.append(sha256_sequence(query_ids))
