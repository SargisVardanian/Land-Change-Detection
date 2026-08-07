from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import TemporalSigLIPConfig


class _TemporalBlock(nn.Module):
    def __init__(self, config: TemporalSigLIPConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * config.mlp_ratio, config.hidden_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.norm_attn(hidden)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden = hidden + attended
        return hidden + self.mlp(self.norm_mlp(hidden))


@dataclass(frozen=True)
class TemporalPairOutput:
    pair_embedding: Tensor
    temporal_tokens: Tensor
    pair_token: Tensor
    frame_count: int
    patch_count: int


class TemporalPairEncoder(nn.Module):
    """Two-block joint temporal transformer over native SigLIP2 patch tokens."""

    def __init__(self, config: TemporalSigLIPConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.pair_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.temporal_embeddings = nn.Parameter(
            torch.empty(config.max_frames, config.hidden_size)
        )
        self.delta_time_projection = nn.Linear(2, config.hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [_TemporalBlock(config) for _ in range(config.temporal_layers)]
        )
        nn.init.normal_(self.pair_token, std=0.02)
        nn.init.normal_(self.temporal_embeddings, std=0.02)
        nn.init.zeros_(self.delta_time_projection.weight)

    def _time_features(
        self,
        timestamps: Tensor | None,
        batch: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if timestamps is None:
            return torch.zeros((batch, frames, self.config.hidden_size), device=device, dtype=dtype)
        if timestamps.shape != (batch, frames):
            raise ValueError("timestamps must have shape [B,T]")
        first = timestamps[:, :1]
        delta = timestamps - first
        scale = delta.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        features = torch.stack((delta, delta / scale), dim=-1)
        return self.delta_time_projection(
            features.to(dtype=self.delta_time_projection.weight.dtype)
        ).to(dtype=dtype)

    def forward(
        self,
        frame_tokens: Tensor,
        *,
        timestamps: Tensor | None = None,
    ) -> TemporalPairOutput:
        if frame_tokens.ndim != 4:
            raise ValueError("frame_tokens must have shape [B,T,N,D]")
        batch, frames, patches, hidden = frame_tokens.shape
        if frames < 2 or frames > self.config.max_frames:
            raise ValueError("frame count is outside the configured range")
        if patches != self.config.patch_tokens or hidden != self.config.hidden_size:
            raise ValueError("native SigLIP2 token shape does not match config")
        time = self._time_features(
            timestamps,
            batch,
            frames,
            frame_tokens.device,
            frame_tokens.dtype,
        )
        temporal = frame_tokens + self.temporal_embeddings[:frames].view(1, frames, 1, hidden)
        temporal = temporal + time.view(batch, frames, 1, hidden)
        flattened = temporal.reshape(batch, frames * patches, hidden)
        pooled_seed = flattened.mean(dim=1, keepdim=True)
        pair = pooled_seed + self.pair_token.to(dtype=frame_tokens.dtype)
        hidden_states = torch.cat((pair, flattened), dim=1)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        pair_embedding = F.normalize(hidden_states[:, 0], dim=-1)
        temporal_tokens = hidden_states[:, 1:].reshape(batch, frames, patches, hidden)
        return TemporalPairOutput(
            pair_embedding=pair_embedding,
            temporal_tokens=temporal_tokens,
            pair_token=hidden_states[:, :1],
            frame_count=frames,
            patch_count=patches,
        )
