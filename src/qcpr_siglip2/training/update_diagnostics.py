"""Optional, non-invasive optimizer update diagnostics.

The diagnostics are intentionally disabled by default.  When enabled by a
training run, parameters are snapshotted before the optimizer step so the
actual update norm can be measured after AdamW (including its moments and
weight decay).  No optimizer or clipping behavior is changed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn


def parameter_diagnostic_group(name: str) -> str:
    """Map model parameter names to stable scientific reporting groups."""

    if name.startswith("temporal_adapter."):
        return "temporal_adapter"
    if name.startswith(("backbone.vision_model.", "backbone.model.vision_model.")):
        return "unfrozen_vision_blocks"
    if name.startswith(("backbone.text_model.", "backbone.model.text_model.")):
        return "unfrozen_text_blocks"
    if name == "log_temperature":
        return "logit_scale"
    return "other_trainable"


def snapshot_trainable_parameters(model: nn.Module) -> dict[str, Tensor]:
    """Clone trainable parameters for optional post-step update accounting."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def optimizer_update_diagnostics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    before_parameters: dict[str, Tensor],
    preclip_gradients: dict[str, Tensor | None],
    postclip_gradients: dict[str, Tensor | None],
    *,
    total_gradient_norm_preclip: float,
    clip_norm: float,
) -> dict[str, Any]:
    """Return per-group gradient, clipping, LR and actual update statistics."""

    named_parameters = dict(model.named_parameters())
    parameter_lrs: dict[int, float] = {}
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        for parameter in group["params"]:
            parameter_lrs[id(parameter)] = lr

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, parameter in named_parameters.items():
        if parameter.requires_grad and name in before_parameters:
            grouped[parameter_diagnostic_group(name)].append(name)

    clip_coefficient = min(
        1.0,
        float(clip_norm) / (float(total_gradient_norm_preclip) + 1e-6),
    )
    clipped = clip_coefficient < 1.0
    result: dict[str, Any] = {
        "enabled": True,
        "gradient_clip_norm": float(clip_norm),
        "total_gradient_norm_preclip": float(total_gradient_norm_preclip),
        "global_clip_coefficient": float(clip_coefficient),
        "global_clipped": bool(clipped),
        "groups": {},
    }

    for group_name, names in sorted(grouped.items()):
        pre_sq = 0.0
        post_sq = 0.0
        update_sq = 0.0
        parameter_before_sq = 0.0
        parameter_after_sq = 0.0
        pre_nonzero = 0
        reduced_elements = 0
        learning_rates: set[float] = set()
        parameter_count = 0

        for name in names:
            parameter = named_parameters[name]
            parameter_count += parameter.numel()
            learning_rates.add(parameter_lrs.get(id(parameter), 0.0))
            before = before_parameters[name].to(device=parameter.device)
            after = parameter.detach()
            parameter_before_sq += float(before.float().square().sum())
            parameter_after_sq += float(after.float().square().sum())
            update_sq += float((after.float() - before.float()).square().sum())

            pre = preclip_gradients.get(name)
            post = postclip_gradients.get(name)
            if pre is None or post is None:
                continue
            pre_float = pre.float()
            post_float = post.float()
            pre_sq += float(pre_float.square().sum())
            post_sq += float(post_float.square().sum())
            nonzero = pre_float.abs() > 1e-12
            pre_nonzero += int(nonzero.sum())
            reduced_elements += int(
                ((post_float.abs() + 1e-12) < pre_float.abs()).logical_and(nonzero).sum()
            )

        parameter_norm_before = parameter_before_sq**0.5
        result["groups"][group_name] = {
            "parameter_count": parameter_count,
            "learning_rate": min(learning_rates) if learning_rates else 0.0,
            "learning_rate_min": min(learning_rates) if learning_rates else 0.0,
            "learning_rate_max": max(learning_rates) if learning_rates else 0.0,
            "gradient_norm_preclip": pre_sq**0.5,
            "gradient_norm_postclip": post_sq**0.5,
            "clip_fraction": (
                float(reduced_elements) / float(pre_nonzero) if pre_nonzero else 0.0
            ),
            "parameter_norm_before": parameter_norm_before,
            "parameter_norm_after": parameter_after_sq**0.5,
            "update_norm": update_sq**0.5,
            "update_weight_ratio": (
                update_sq**0.5 / max(parameter_norm_before, 1e-12)
            ),
        }

    return result
