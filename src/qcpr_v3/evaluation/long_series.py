"""Tensor-only long-series contract checks until real data is enabled."""

from __future__ import annotations

import torch
from torch import Tensor


def reverse_temporal_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim < 2:
        raise ValueError("tokens must have a temporal dimension")
    return tokens.flip(dims=(1,))


def pad_temporal_tokens(sequences: list[Tensor], pad_value: float = 0.0) -> tuple[Tensor, Tensor]:
    if not sequences:
        raise ValueError("sequences must be non-empty")
    if any(item.ndim != sequences[0].ndim for item in sequences):
        raise ValueError("sequence ranks must match")
    max_time = max(item.shape[0] for item in sequences)
    result = sequences[0].new_full((len(sequences), max_time, *sequences[0].shape[1:]), pad_value)
    mask = torch.zeros((len(sequences), max_time), dtype=torch.bool, device=result.device)
    for row, sequence in enumerate(sequences):
        result[row, : sequence.shape[0]] = sequence
        mask[row, : sequence.shape[0]] = True
    return result, mask
