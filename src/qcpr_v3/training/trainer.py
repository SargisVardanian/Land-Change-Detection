"""Canonical one-loss training step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.model import QCPRV3Model
from ..models.temporal_adapter import TemporalOutput
from ..models.jina_query_encoder import QueryFeatures
from .objective import LossOutput, UnifiedListwiseLoss


@dataclass(frozen=True)
class TrainStepOutput:
    scores: Tensor
    loss: LossOutput
    evidence_gradient_norm: float


def train_step(
    model: QCPRV3Model,
    query: QueryFeatures,
    visual: TemporalOutput,
    grades: Tensor,
    *,
    objective: UnifiedListwiseLoss,
    valid_mask: Tensor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainStepOutput:
    model.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    scored = model.score(query, visual)
    loss = objective(scored.scores, grades, valid_mask)
    if optimizer is not None:
        loss.loss.backward()
        grad = model.evidence_bottleneck.query_projection.weight.grad
        evidence_gradient_norm = 0.0 if grad is None else float(grad.detach().norm())
        optimizer.step()
    else:
        evidence_gradient_norm = 0.0
    if not torch.isfinite(loss.loss):
        raise FloatingPointError("unified retrieval loss is not finite")
    return TrainStepOutput(scored.scores, loss, evidence_gradient_norm)
