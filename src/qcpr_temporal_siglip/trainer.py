from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .model import TemporalSigLIP
from .objective import symmetric_mult_positive_clip_loss


_LAYER_RE = re.compile(r"\.encoder\.layers\.(\d+)")


@dataclass(frozen=True)
class TrainStepResult:
    loss: Tensor
    gradient_norm_preclip: float
    score_matrix: Tensor


def _no_decay(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith(".bias")
        or "norm" in lowered
        or "embedding" in lowered
        or "pair_token" in lowered
        or "logit_scale" in lowered
    )


def build_optimizer(
    model: TemporalSigLIP,
    *,
    stage: str,
    total_steps: int,
    warmup_fraction: float = 0.05,
    weight_decay: float = 0.05,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict[str, Any]]:
    """Build a fresh optimizer for Stage A or Stage B."""

    if stage not in {"A", "B"}:
        raise ValueError("stage must be A or B")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if stage == "A":
        model.set_stage_a()
        if model.backbone is not None and any(
            parameter.requires_grad for parameter in model.backbone.parameters()
        ):
            raise RuntimeError("Stage A requires frozen SigLIP2 towers")
    else:
        model.set_stage_b()
    buckets: dict[tuple[str, float, float], list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("temporal_pair."):
            group, lr = "temporal_pair", 1e-4
        elif name == "logit_scale":
            group, lr = "logit_scale", 1e-5
        elif "vision_model.encoder.layers." in name:
            group, lr = "vision_last_two_blocks", 5e-6
        elif "text_model.encoder.layers." in name:
            group, lr = "text_last_two_blocks", 5e-6
        elif any(token in name for token in ("post_layernorm", "final_layer_norm", ".head")):
            group, lr = "pretrained_final_norms_projections", 2e-5
        else:
            raise RuntimeError(f"unapproved trainable parameter: {name}")
        buckets[(group, lr, 0.0 if _no_decay(name) else weight_decay)].append((name, parameter))
    if not buckets:
        raise RuntimeError("no trainable parameters found")
    parameter_groups: list[dict[str, Any]] = []
    report_groups: list[dict[str, Any]] = []
    for (group, lr, decay), named in sorted(buckets.items()):
        params = [parameter for _, parameter in named]
        parameter_groups.append({"params": params, "lr": lr, "weight_decay": decay})
        report_groups.append({
            "name": group,
            "lr": lr,
            "weight_decay": decay,
            "parameter_count": sum(parameter.numel() for parameter in params),
            "parameter_names": [name for name, _ in named],
        })
    optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.98), eps=1e-6)
    warmup_steps = round(total_steps * warmup_fraction)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0), remaining) / remaining
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    report = {
        "stage": stage,
        "total_steps": total_steps,
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "optimizer": "AdamW",
        "weight_decay_default": weight_decay,
        "groups": report_groups,
        "trainable_parameter_count": sum(item["parameter_count"] for item in report_groups),
        "new_optimizer_at_stage_transition": stage == "B",
    }
    return optimizer, scheduler, report


def train_feature_step(
    model: TemporalSigLIP,
    frame_tokens: Tensor,
    text_tokens: Tensor,
    text_embeddings: Tensor,
    text_mask: Tensor,
    positive_mask: Tensor,
    ignored_mask: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    timestamps: Tensor | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    gradient_clip_norm: float = 1.0,
) -> TrainStepResult:
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_from_features(
        frame_tokens,
        text_tokens,
        text_embeddings,
        text_mask,
        timestamps=timestamps,
    )
    loss = symmetric_mult_positive_clip_loss(
        output.score_matrix,
        positive_mask,
        ignored_mask,
    )
    loss.backward()
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("TRAINABLE_MODULE_NO_GRADIENT")
    flat = torch.cat(gradients)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("NONFINITE_GRADIENT")
    norm = float(flat.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return TrainStepResult(loss.detach(), norm, output.score_matrix.detach())
