from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SequenceVisualFeatures:
    features: Tensor
    grid_height: int
    grid_width: int
    metadata: dict[str, Any]


class SequenceUniverSatEncoder(nn.Module):
    """Per-timestamp UniverSat wrapper for UniChange v2.

    The wrapper deliberately flattens `[B,T,C,H,W]` to `[B*T,C,H,W]` so the
    temporal axis remains explicit outside UniverSat. Native temporal UniverSat
    encoding can be compared later as a separate baseline.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        output_grid: int = 32,
        visual_dim: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.output_grid = output_grid
        self.visual_dim = visual_dim
        self.freeze = freeze
        if freeze:
            self.image_encoder.eval()
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad_(False)

    def _normalize_output(self, output: Any) -> Tensor:
        if isinstance(output, dict):
            for key in ("tokens", "last_hidden_state", "features", "x"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    output = value
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError(f"UniverSat output must resolve to a Tensor, got {type(output)!r}.")
        if output.ndim == 4:
            output = output.flatten(2).transpose(1, 2)
        if output.ndim != 3:
            raise ValueError(f"Expected per-frame tokens [B*T,N,D], got {tuple(output.shape)}.")
        return output

    def forward(self, images: Tensor) -> SequenceVisualFeatures:
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B,T,C,H,W], got {tuple(images.shape)}")
        batch_size, time_steps, channels, height, width = images.shape
        flat = images.reshape(batch_size * time_steps, channels, height, width)
        with torch.set_grad_enabled(not self.freeze):
            try:
                output = self.image_encoder(flat, output_grid=self.output_grid)
            except TypeError:
                output = self.image_encoder(flat)
        tokens = self._normalize_output(output)
        expected_tokens = self.output_grid * self.output_grid
        if tokens.shape[1] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} spatial tokens, got {tokens.shape[1]}.")
        if tokens.shape[-1] != self.visual_dim:
            raise ValueError(f"Expected visual dim {self.visual_dim}, got {tokens.shape[-1]}.")
        return SequenceVisualFeatures(
            features=tokens.reshape(batch_size, time_steps, expected_tokens, self.visual_dim),
            grid_height=self.output_grid,
            grid_width=self.output_grid,
            metadata={
                "backend": "sequence_universat_per_timestamp",
                "input_shape": [batch_size, time_steps, channels, height, width],
                "output_grid": self.output_grid,
                "frozen": self.freeze,
            },
        )
