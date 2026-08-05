"""Deterministic exact-core sampling without pair-weight drift."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .manifest import group_rows_by_pair


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class ExactBatch:
    pair_rows: tuple[dict[str, Any], ...]
    query_rows: tuple[dict[str, Any], ...]
    pair_ids: tuple[str, ...]
    query_ids: tuple[str, ...]


def select_epoch_rows(
    rows: Iterable[dict[str, Any]],
    *,
    captions_per_pair: int = 2,
    epoch: int = 0,
    seed: int = 20260805,
) -> list[dict[str, Any]]:
    """Rotate caption windows while preserving one equal pair weight."""

    if captions_per_pair <= 0:
        raise ValueError("captions_per_pair must be positive")
    selected: list[dict[str, Any]] = []
    for pair_id, group in group_rows_by_pair(rows).items():
        if len(group) < captions_per_pair:
            raise ValueError(
                f"pair {pair_id} has fewer than {captions_per_pair} captions"
            )
        offset = (epoch + _stable_int(pair_id) + seed) % len(group)
        for index in range(captions_per_pair):
            selected.append(group[(offset + index) % len(group)])
    return selected


def make_exact_batches(
    rows: Iterable[dict[str, Any]],
    *,
    physical_batch_size: int,
    captions_per_pair: int = 2,
    epoch: int = 0,
    seed: int = 20260805,
) -> list[ExactBatch]:
    """Make deterministic batches; the same pair ordering drives all arms."""

    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    selected = select_epoch_rows(
        rows, captions_per_pair=captions_per_pair, epoch=epoch, seed=seed
    )
    grouped = group_rows_by_pair(selected)
    pair_ids = list(grouped)
    random.Random(seed + epoch).shuffle(pair_ids)
    batches: list[ExactBatch] = []
    for start in range(0, len(pair_ids) - physical_batch_size + 1, physical_batch_size):
        batch_ids = pair_ids[start : start + physical_batch_size]
        query_rows = tuple(row for pair_id in batch_ids for row in grouped[pair_id])
        pair_rows = tuple(grouped[pair_id][0] for pair_id in batch_ids)
        batches.append(
            ExactBatch(
                pair_rows=pair_rows,
                query_rows=query_rows,
                pair_ids=tuple(batch_ids),
                query_ids=tuple(str(row["caption_id"]) for row in query_rows),
            )
        )
    return batches
