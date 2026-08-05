"""Explicit Phase-A/Phase-B optimizer scopes for the minimal track."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch
from torch import nn

from ..models.model import Siglip2TemporalRetrievalModel

_LAYER_RE = re.compile(r"(?:vision_model|text_model)\.encoder\.layers\.(\d+)")


def _no_decay(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith(".bias")
        or "layernorm" in lowered
        or ".norm" in lowered
        or "frame_position" in lowered
        or "frame_type" in lowered
        or "patch_type" in lowered
        or "spatial_position" in lowered
        or "time_projection" in lowered
        or "raw_gate" in lowered
        or "log_temperature" in lowered
    )


def _group_name(name: str, phase: str, top_blocks: int) -> tuple[str, float]:
    if name.startswith("temporal_adapter"):
        return "temporal_adapter", 1e-4
    if name.startswith("evidence_bottleneck"):
        return "evidence", 1e-5
    if name == "log_temperature":
        return "retrieval_temperature", 1e-5
    if phase == "A":
        raise RuntimeError(f"unexpected Phase-A trainable parameter: {name}")
    match = _LAYER_RE.search(name)
    if match is not None:
        if "vision_model" in name:
            return "vision_top_blocks", 5e-6
        if "text_model" in name:
            return "text_top_blocks", 5e-6
        raise RuntimeError(f"unrecognized pretrained block: {name}")
    if any(
        token in name.lower()
        for token in ("post_layernorm", "final_layer_norm", ".head", "text_projection")
    ):
        return "pretrained_final_norms_and_heads", 2e-5
    raise RuntimeError(f"unapproved Phase-B trainable parameter: {name}")


def build_adamw(
    model: Siglip2TemporalRetrievalModel,
    *,
    phase: str,
    total_steps: int,
    warmup_fraction: float = 0.05,
    weight_decay: float = 0.05,
) -> tuple[torch.optim.Optimizer, dict[str, Any], torch.optim.lr_scheduler.LambdaLR]:
    """Build a fresh optimizer and scheduler with an auditable parameter scope."""

    if phase not in {"A", "B"}:
        raise ValueError("phase must be A or B")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    top_blocks = (
        int(getattr(model.backbone, "phase_b_top_blocks", 0))
        if model.backbone is not None
        else 0
    )
    if phase == "B" and top_blocks != 2:
        raise ValueError("Phase B requires exactly two enabled top backbone blocks")
    if (
        phase == "A"
        and model.backbone is not None
        and any(parameter.requires_grad for parameter in model.backbone.parameters())
    ):
        raise ValueError("Phase A requires a fully frozen pretrained backbone")

    buckets: dict[tuple[str, float, float], list[tuple[str, nn.Parameter]]] = (
        defaultdict(list)
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group, learning_rate = _group_name(name, phase, top_blocks)
        decay = 0.0 if _no_decay(name) else weight_decay
        buckets[(group, learning_rate, decay)].append((name, parameter))
    if not buckets:
        raise RuntimeError("no trainable parameters found for optimizer")

    parameter_groups: list[dict[str, Any]] = []
    report_groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for (group, learning_rate, decay), named_parameters in sorted(buckets.items()):
        parameters = [parameter for _, parameter in named_parameters]
        overlap = {id(parameter) for parameter in parameters} & seen
        if overlap:
            raise RuntimeError(
                f"parameter appears in multiple optimizer groups: {group}"
            )
        seen.update(id(parameter) for parameter in parameters)
        parameter_groups.append(
            {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        )
        report_groups.append(
            {
                "name": group,
                "lr": learning_rate,
                "weight_decay": decay,
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "parameter_names": [name for name, _ in named_parameters],
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.98), eps=1e-6)
    warmup_steps = round(total_steps * warmup_fraction)

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0), remaining) / remaining
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    report = {
        "phase": phase,
        "total_steps": total_steps,
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "optimizer": "AdamW",
        "betas": [0.9, 0.98],
        "eps": 1e-6,
        "weight_decay_default": weight_decay,
        "groups": report_groups,
        "trainable_parameter_count": sum(
            item["parameter_count"] for item in report_groups
        ),
    }
    return optimizer, report, scheduler
