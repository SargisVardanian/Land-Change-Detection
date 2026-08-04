"""Gradient diagnostics without hidden loss branches."""

from __future__ import annotations

from torch import nn


def gradient_norms(module: nn.Module) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            result[name] = float(parameter.grad.detach().norm())
    return result
