"""One-loss training primitives shared by bounded SigLIP-2 pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..models.model import RetrievalForwardOutput, Siglip2TemporalRetrievalModel
from .objective import symmetric_multi_positive_listwise_loss


@dataclass(frozen=True)
class FeatureBatch:
    frame_tokens: Tensor
    frame_embeddings: Tensor
    text_tokens: Tensor
    text_embeddings: Tensor
    text_mask: Tensor
    positive_mask: Tensor
    ignored_mask: Tensor
    timestamps: Tensor | None = None


@dataclass(frozen=True)
class StepResult:
    loss: Tensor
    output: RetrievalForwardOutput
    gradient_norm_preclip: float


def train_feature_step(
    model: Siglip2TemporalRetrievalModel,
    batch: FeatureBatch,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    gradient_clip_norm: float = 1.0,
) -> StepResult:
    """Run exactly one differentiable query-to-pair listwise step."""

    optimizer.zero_grad(set_to_none=True)
    output = model.forward_from_features(
        batch.frame_tokens,
        batch.frame_embeddings,
        batch.text_tokens,
        batch.text_embeddings,
        batch.text_mask,
        timestamps=batch.timestamps,
    )
    loss = symmetric_multi_positive_listwise_loss(
        output.score_matrix.float(), batch.positive_mask, batch.ignored_mask
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("primary listwise loss is non-finite")
    loss.backward()
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("TRAINABLE_MODULE_NO_GRADIENT")
    flat = torch.cat(gradients)
    if not torch.isfinite(flat).all():
        raise FloatingPointError("NONFINITE_GRADIENT")
    gradient_norm = float(flat.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return StepResult(loss.detach(), output, gradient_norm)


def checkpoint_state(
    model: Siglip2TemporalRetrievalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    global_step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a safe tensor/primitives-only checkpoint payload."""

    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": int(global_step),
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state
