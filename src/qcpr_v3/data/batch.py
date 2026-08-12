"""Logical batch construction and deterministic exposure hashes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .contracts import QueryRecord, RetrievalItem


def sequence_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LogicalBatch:
    pair_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    grades: Tensor
    valid_mask: Tensor

    def validate(self) -> None:
        if self.grades.shape != self.valid_mask.shape:
            raise ValueError("grades and valid_mask must have the same shape")
        if self.grades.shape != (len(self.query_ids), len(self.pair_ids)):
            raise ValueError("logical batch tensors do not match ordered IDs")


def build_relevance_matrix(
    queries: Sequence[QueryRecord],
    items: Sequence[RetrievalItem],
) -> tuple[Tensor, Tensor]:
    item_index = {item.item_id: index for index, item in enumerate(items)}
    if len(item_index) != len(items):
        raise ValueError("duplicate item IDs in logical gallery")
    grades = torch.zeros((len(queries), len(items)), dtype=torch.long)
    valid = torch.ones_like(grades, dtype=torch.bool)
    for row, query in enumerate(queries):
        for item_id, grade in query.graded_relevance.items():
            if item_id not in item_index:
                raise ValueError(f"query {query.query_id} references unknown item {item_id}")
            grades[row, item_index[item_id]] = int(grade)
        for item_id in query.positive_item_ids:
            if item_id not in item_index:
                raise ValueError(f"query {query.query_id} references unknown positive {item_id}")
            grades[row, item_index[item_id]] = max(1, grades[row, item_index[item_id]])
    return grades, valid


def schedule_hash(pair_ids: Sequence[str], query_ids: Sequence[str]) -> dict[str, str]:
    return {
        "pair_sequence_sha256": sequence_sha256(pair_ids),
        "query_sequence_sha256": sequence_sha256(query_ids),
    }


def assert_equal_schedule(first: dict[str, str], second: dict[str, str]) -> None:
    required = {"pair_sequence_sha256", "query_sequence_sha256"}
    if not required.issubset(first) or not required.issubset(second):
        raise ValueError("schedule hashes are incomplete")
    if first != second:
        raise ValueError("exposure schedule hashes differ")
