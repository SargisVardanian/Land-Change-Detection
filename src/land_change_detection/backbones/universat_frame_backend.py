from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class UniverSatFrameBackend(nn.Module):
    def __init__(self, model: nn.Module, output_grid: int = 32, visual_dim: int = 768):
        super().__init__()
        self.model = model
        self.output_grid = output_grid
        self.visual_dim = visual_dim
        if not callable(getattr(model, "encode", None)):
            raise TypeError("UniverSat model must expose encode().")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, frames: Tensor, output_grid: int | None = None) -> Tensor:
        grid = int(output_grid or self.output_grid)
        if frames.ndim != 4:
            raise ValueError("frames must have shape [B,C,H,W]")
        with torch.no_grad():
            output: Any = self.model.encode({"spot": frames}, patch_size=10.0, output_grid=grid)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError("UniverSat encode() did not return tensor tokens")
        if output.ndim == 4 and output.shape[-1] == self.visual_dim:
            output = output.reshape(output.shape[0], -1, self.visual_dim)
        if output.ndim != 3 or output.shape[1:] != (grid * grid, self.visual_dim):
            raise ValueError(f"Unexpected UniverSat token shape: {tuple(output.shape)}")
        return output
