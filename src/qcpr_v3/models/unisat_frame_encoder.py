"""Frozen UniverSat frame encoder adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.contracts import TemporalMetadata, VisualTokenBatch


@dataclass(frozen=True)
class FrameEncoderContract:
    model_identifier: str
    revision: str
    checkpoint_sha256: str
    native_dim: int
    token_shape: tuple[int, int]
    native_stride: int | None
    stage_names: tuple[str, ...]
    frozen_parameters: int
    trainable_parameters: int


class FrozenFrameEncoder(nn.Module):
    """Apply one shared frozen frame encoder at every timestamp."""

    def __init__(
        self,
        backend: nn.Module,
        *,
        native_dim: int,
        contract: FrameEncoderContract | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.native_dim = native_dim
        self.contract = contract
        for parameter in self.backend.parameters():
            parameter.requires_grad_(False)
        self.backend.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backend.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        frames: Tensor,
        metadata: TemporalMetadata,
        coordinates: Tensor,
        *,
        frame_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> VisualTokenBatch:
        if frames.ndim != 5:
            raise ValueError("frames must be [B,T,C,H,W]")
        batch, time_count = frames.shape[:2]
        encoded: list[Tensor] = []
        for frame_index in range(time_count):
            tokens = self.backend(frames[:, frame_index])
            if tokens.ndim != 3 or tokens.shape[0] != batch or tokens.shape[-1] != self.native_dim:
                raise ValueError("frame backend must return [B,N,native_dim]")
            encoded.append(tokens)
        dense = torch.stack(encoded, dim=1)
        n = dense.shape[2]
        if coordinates.shape != (batch, n, 2):
            raise ValueError("coordinates do not match native token count")
        frame_mask = (
            torch.ones((batch, time_count), dtype=torch.bool, device=frames.device)
            if frame_mask is None
            else frame_mask.to(dtype=torch.bool, device=frames.device)
        )
        token_mask = (
            torch.ones((batch, time_count, n), dtype=torch.bool, device=frames.device)
            if token_mask is None
            else token_mask.to(dtype=torch.bool, device=frames.device)
        )
        result = VisualTokenBatch(dense, coordinates, frame_mask, token_mask, metadata)
        result.validate()
        return result
