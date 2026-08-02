"""Deterministic source/domain/event/change balanced sampling for QCPR Stage-2."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class HierarchicalSamplingConfig:
    batch_size: int = 128
    max_event_fraction: float = 0.125
    max_source_fraction: float = 0.50
    seed: int = 20260803


def _source(row: dict[str, Any]) -> str:
    return str(row.get("source_dataset") or row.get("source") or "unknown")


def _domain(row: dict[str, Any]) -> str:
    explicit = row.get("domain") or row.get("disaster_domain")
    if explicit:
        return str(explicit)
    return "disaster" if row.get("source_event_id") or "event_id" in row else "non_disaster"


def _event(row: dict[str, Any]) -> str:
    return str(row.get("source_event_id") or row.get("event_id") or row.get("source_scene_group_id") or "unknown_event")


def _change(row: dict[str, Any]) -> str:
    return str(row.get("change_type") or row.get("change_category") or row.get("query_scope") or "unspecified")


class HierarchicalSourceSampler:
    """Sample physical rows under source -> domain -> event -> change caps.

    Rows are never duplicated inside one batch. `sample_epoch` balances event
    presentations across the epoch while preserving the configured caps.
    """

    def __init__(self, rows: Iterable[dict[str, Any]], config: HierarchicalSamplingConfig = HierarchicalSamplingConfig()):
        source_rows = list(rows)
        self.rows: list[dict[str, Any]] = []
        self.config = config
        if not source_rows:
            raise ValueError("empty sampler")
        if config.batch_size <= 0 or not (0 < config.max_event_fraction <= 1):
            raise ValueError("invalid batch or event fraction")
        self.by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_hierarchy: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.row_ids: set[str] = set()
        for i, row in enumerate(source_rows):
            pair_id = str(row.get("canonical_pair_id") or row.get("pair_id") or f"row-{i}")
            if pair_id in self.row_ids:
                raise ValueError(f"duplicate physical pair: {pair_id}")
            self.row_ids.add(pair_id)
            source, domain, event, change = _source(row), _domain(row), _event(row), _change(row)
            item = dict(row)
            item["canonical_pair_id"] = pair_id
            item["_source"] = source
            item["_domain"] = domain
            item["_event"] = event
            item["_change"] = change
            self.rows.append(item)
            self.by_event[event].append(item)
            self.by_source[source].append(item)
            self.by_hierarchy[(source, domain, event, change)].append(item)
        self.events = sorted(self.by_event)

    @property
    def max_event_count(self) -> int:
        return max(1, int(self.config.batch_size * self.config.max_event_fraction))

    @property
    def max_source_count(self) -> int:
        return max(1, int(self.config.batch_size * self.config.max_source_fraction))

    def _ordered(self, rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
        result = list(rows)
        rng.shuffle(result)
        return result

    def sample_batch(self, batch_index: int = 0, target_event_counts: Counter[str] | None = None) -> list[dict[str, Any]]:
        rng = random.Random(self.config.seed + batch_index * 1009)
        event_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        candidates = self._ordered(self.rows, rng)
        if target_event_counts is None:
            target_event_counts = Counter()
        # First touch the least represented events so a batch cannot collapse
        # onto one source/event when a diverse pool is available.
        candidates.sort(key=lambda r: (target_event_counts[_event(r)], event_counts[_event(r)], _event(r), str(r["canonical_pair_id"])))
        while candidates and len(selected) < self.config.batch_size:
            progressed = False
            for row in candidates:
                pair_id = row["canonical_pair_id"]
                event, source = row["_event"], row["_source"]
                if pair_id in used or event_counts[event] >= self.max_event_count or source_counts[source] >= self.max_source_count:
                    continue
                used.add(pair_id); selected.append(row); event_counts[event] += 1; source_counts[source] += 1; progressed = True
                if len(selected) >= self.config.batch_size:
                    break
            if not progressed:
                break
        if len(selected) < min(self.config.batch_size, len(self.rows)):
            raise ValueError("sampling constraints cannot fill the requested batch")
        return selected

    def sample_epoch(self, num_batches: int) -> list[list[dict[str, Any]]]:
        if num_batches <= 0:
            return []
        epoch_counts: Counter[str] = Counter()
        batches = []
        for batch_index in range(num_batches):
            batch = self.sample_batch(batch_index, epoch_counts)
            batches.append(batch)
            epoch_counts.update(_event(row) for row in batch)
        return batches

    @staticmethod
    def counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(_event(row) for row in rows))

    @staticmethod
    def schedule_sha256(batches: Iterable[Iterable[dict[str, Any]]]) -> str:
        lines = []
        for batch_index, batch in enumerate(batches):
            for row in batch:
                lines.append(f"{batch_index}:{row.get('canonical_pair_id') or row.get('pair_id')}:{_source(row)}:{_domain(row)}:{_event(row)}:{_change(row)}")
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
