"""Two-stage ANN candidate selection and query-conditioned reranking."""

from __future__ import annotations

import torch
from torch import Tensor


def _has_row_duplicates(values: Tensor) -> bool:
    if values.shape[1] < 2:
        return False
    sorted_values = values.sort(dim=1).values
    return bool((sorted_values[:, 1:] == sorted_values[:, :-1]).any())


def select_topk_candidates(global_scores: Tensor, k: int) -> Tensor:
    if global_scores.ndim != 2 or k <= 0:
        raise ValueError("global_scores must be [Q,P] and k must be positive")
    return global_scores.topk(
        min(k, global_scores.shape[1]), dim=1, largest=True, sorted=True
    ).indices


def rerank_candidate_indices(
    candidate_indices: Tensor, rerank_scores: Tensor
) -> Tensor:
    if candidate_indices.ndim != 2 or rerank_scores.shape != candidate_indices.shape:
        raise ValueError("candidate indices and rerank scores must share [Q,K] shape")
    if _has_row_duplicates(candidate_indices):
        raise ValueError("candidate list contains duplicate physical items")
    order = rerank_scores.argsort(dim=1, descending=True)
    return candidate_indices.gather(1, order)


def scatter_reranked_scores(
    global_scores: Tensor, candidate_indices: Tensor, rerank_scores: Tensor
) -> Tensor:
    if (
        global_scores.ndim != 2
        or candidate_indices.ndim != 2
        or rerank_scores.shape != candidate_indices.shape
    ):
        raise ValueError("invalid score/index shapes")
    if torch.any(candidate_indices < 0) or torch.any(
        candidate_indices >= global_scores.shape[1]
    ):
        raise ValueError("candidate index outside gallery")
    if _has_row_duplicates(candidate_indices):
        raise ValueError("candidate list contains duplicate physical items")
    result = global_scores.clone()
    result.scatter_(1, candidate_indices, rerank_scores)
    return result
