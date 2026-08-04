"""Small explicit checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    metadata: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "metadata": dict(metadata),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
