"""Evaluation-only diagnostics for causal evidence maps."""

from __future__ import annotations

import torch
from torch import Tensor


def evidence_diagnostics(weights: Tensor) -> dict[str, float]:
    if weights.ndim < 2:
        raise ValueError("weights must include a token dimension")
    flat = weights.reshape(weights.shape[0], -1).clamp_min(0.0)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(flat.clamp_min(1e-8) * flat.clamp_min(1e-8).log()).sum(dim=1)
    normalized_entropy = entropy / torch.log(torch.tensor(float(flat.shape[1]), device=flat.device)).clamp_min(1e-8)
    effective = 1.0 / flat.square().sum(dim=1).clamp_min(1e-8)
    return {
        "normalized_entropy": float(normalized_entropy.mean()),
        "effective_patch_count": float(effective.mean()),
        "max_mass": float(flat.max(dim=1).values.mean()),
    }


def query_swap_sensitivity(first: Tensor, second: Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError("maps must share shape")
    return float((first - second).abs().mean())


def deletion_score_drop(full_score: Tensor, deleted_score: Tensor) -> float:
    if full_score.shape != deleted_score.shape:
        raise ValueError("scores must share shape")
    return float((full_score - deleted_score).mean())
