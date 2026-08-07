from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import TemporalSigLIP


def build_checkpoint(
    model: TemporalSigLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    global_step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": int(global_step),
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    path: str | Path,
    model: TemporalSigLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    global_step: int,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step=global_step,
            metadata=metadata,
        ),
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: TemporalSigLIP,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("invalid TemporalSigLIP checkpoint")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if "scheduler_state" not in payload:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload
