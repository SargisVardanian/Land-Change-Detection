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
    # Local physical-pair index for every unique query row.  The sequence is
    # ordered by pair and therefore also defines the variable-Q chunk offsets.
    query_pair_indices: tuple[int, ...] = ()


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
        if not group:
            raise ValueError(
                f"pair {pair_id} has no captions"
            )
        # The exact r19g core contains trusted physical pairs with only one
        # human caption.  Reuse that verified row deterministically rather
        # than dropping the physical item or fabricating a paraphrase.  The
        # runtime contract records this reuse explicitly.
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


def select_unique_epoch_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_unique_captions_per_pair: int = 2,
    epoch: int = 0,
    seed: int = 20260805,
) -> list[dict[str, Any]]:
    """Select distinct trusted captions without manufacturing query rows.

    The cap is only an upper bound.  A singleton pair contributes exactly one
    query, while a pair with several captions contributes at most the
    configured number of distinct ``caption_id`` values.  Caption rotation is
    deterministic and never repeats a query within one logical batch.
    """

    if max_unique_captions_per_pair <= 0:
        raise ValueError("max_unique_captions_per_pair must be positive")
    selected: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for pair_id, group in group_rows_by_pair(rows).items():
        if not group:
            raise ValueError(f"pair {pair_id} has no captions")
        group_query_ids = [str(row.get("caption_id", "")) for row in group]
        if any(not query_id for query_id in group_query_ids):
            raise ValueError(f"pair {pair_id} has a caption without caption_id")
        if len(set(group_query_ids)) != len(group_query_ids):
            raise ValueError(f"pair {pair_id} contains duplicate caption_id values")
        count = min(max_unique_captions_per_pair, len(group))
        offset = (epoch + _stable_int(str(pair_id)) + seed) % len(group)
        for delta in range(count):
            row = group[(offset + delta) % len(group)]
            query_id = str(row["caption_id"])
            if query_id in seen_query_ids:
                raise ValueError(
                    "QUERY_ID_DUPLICATE_IN_LOGICAL_SELECTION:" + query_id
                )
            seen_query_ids.add(query_id)
            selected.append(row)
    return selected


def make_unique_exact_batches(
    rows: Iterable[dict[str, Any]],
    *,
    physical_batch_size: int,
    max_unique_captions_per_pair: int = 2,
    epoch: int = 0,
    seed: int = 20260805,
) -> list[ExactBatch]:
    """Make deterministic logical batches with variable query count ``Q``."""

    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    selected = select_unique_epoch_rows(
        rows,
        max_unique_captions_per_pair=max_unique_captions_per_pair,
        epoch=epoch,
        seed=seed,
    )
    grouped = group_rows_by_pair(selected)
    pair_ids = list(grouped)
    random.Random(seed + epoch).shuffle(pair_ids)
    batches: list[ExactBatch] = []
    for start in range(0, len(pair_ids) - physical_batch_size + 1, physical_batch_size):
        batch_ids = pair_ids[start : start + physical_batch_size]
        pair_rows = tuple(grouped[pair_id][0] for pair_id in batch_ids)
        query_rows_list: list[dict[str, Any]] = []
        query_pair_indices: list[int] = []
        for pair_index, pair_id in enumerate(batch_ids):
            for row in grouped[pair_id]:
                query_rows_list.append(row)
                query_pair_indices.append(pair_index)
        query_ids = tuple(str(row["caption_id"]) for row in query_rows_list)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("QUERY_ID_DUPLICATE_IN_LOGICAL_BATCH")
        batches.append(
            ExactBatch(
                pair_rows=pair_rows,
                query_rows=tuple(query_rows_list),
                pair_ids=tuple(batch_ids),
                query_ids=query_ids,
                query_pair_indices=tuple(query_pair_indices),
            )
        )
    return batches
