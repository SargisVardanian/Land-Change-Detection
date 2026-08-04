"""Embedding geometry diagnostics."""

from __future__ import annotations

import torch
from torch import Tensor


def effective_rank(embeddings: Tensor, *, tolerance: float = 1e-6) -> float:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N,D]")
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    singular = singular[singular > tolerance * singular.max().clamp_min(tolerance)]
    if singular.numel() == 0:
        return 0.0
    probabilities = singular / singular.sum()
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def embedding_diagnostics(embeddings: Tensor) -> dict[str, float]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N,D]")
    return {
        "count": float(embeddings.shape[0]),
        "dimension": float(embeddings.shape[1]),
        "mean_norm": float(embeddings.norm(dim=-1).mean()),
        "std_norm": float(embeddings.norm(dim=-1).std(unbiased=False)),
        "effective_rank": effective_rank(embeddings),
    }
