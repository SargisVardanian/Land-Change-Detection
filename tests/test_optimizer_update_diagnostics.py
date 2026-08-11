import torch
from torch import nn

from qcpr_siglip2.training.update_diagnostics import (
    optimizer_update_diagnostics,
    snapshot_trainable_parameters,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_adapter = nn.Linear(3, 3, bias=False)
        self.backbone = nn.Module()
        self.backbone.vision_model = nn.Linear(3, 3, bias=False)
        self.backbone.text_model = nn.Linear(3, 3, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.1))


def test_optimizer_update_diagnostics_reports_groups_and_actual_update() -> None:
    torch.manual_seed(11)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()

    before = snapshot_trainable_parameters(model)
    preclip = {
        name: parameter.grad.detach().clone()
        if parameter.grad is not None
        else None
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    postclip = {
        name: parameter.grad.detach().clone()
        if parameter.grad is not None
        else None
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer.step()

    diagnostics = optimizer_update_diagnostics(
        model,
        optimizer,
        before,
        preclip,
        postclip,
        total_gradient_norm_preclip=total,
        clip_norm=1.0,
    )
    groups = diagnostics["groups"]
    assert diagnostics["enabled"] is True
    assert set(groups) == {
        "logit_scale",
        "temporal_adapter",
        "unfrozen_text_blocks",
        "unfrozen_vision_blocks",
    }
    assert diagnostics["global_clip_coefficient"] <= 1.0
    assert all(row["update_norm"] > 0.0 for row in groups.values())
    assert all(row["update_weight_ratio"] > 0.0 for row in groups.values())
    assert all(row["learning_rate"] == 1e-3 for row in groups.values())
