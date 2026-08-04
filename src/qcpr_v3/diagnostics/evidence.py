"""Diagnostics for the causal evidence route."""

from __future__ import annotations

from torch import nn


def evidence_gradient_diagnostics(module: nn.Module) -> dict[str, float | bool]:
    names = ("evidence_bottleneck", "relevance_model.evidence_gate", "relevance_model.late_gate")
    result: dict[str, float | bool] = {}
    for name, parameter in module.named_parameters():
        if any(name.startswith(prefix) for prefix in names):
            result[name] = 0.0 if parameter.grad is None else float(parameter.grad.detach().norm())
    result["nonzero_evidence_gradient"] = any(
        isinstance(value, float) and value > 0.0 for key, value in result.items() if key != "nonzero_evidence_gradient"
    )
    return result
