"""Top-K reranking utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def rerank_candidate_scores(stage_one: Tensor, stage_two: Tensor, candidate_indices: Tensor) -> Tensor:
    if stage_one.ndim != 2 or stage_two.shape != candidate_indices.shape:
        raise ValueError("reranking inputs must be [Q,K]")
    output = stage_one.clone()
    output.scatter_(1, candidate_indices, stage_two)
    return output
