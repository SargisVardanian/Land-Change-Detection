from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalFusionConfig:
    input_dim: int = 768
    model_dim: int = 512
    output_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    ffn_ratio: int = 4
    event_queries: int = 32
    event_decoder_layers: int = 2
    max_time_steps: int = 32
    dropout: float = 0.0


@dataclass(frozen=True)
class TemporalFusionOutput:
    temporal_tokens: Tensor
    change_tokens: Tensor
    event_tokens: Tensor
    global_embedding: Tensor
    temporal_attention: Tensor


class TemporalAxisBlock(nn.Module):
    """Self-attention along time independently for every spatial position."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, temporal_padding_mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B, T, N, D].")
        batch_size, time_steps, num_tokens, dim = tokens.shape
        sequence = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, time_steps, dim)
        normalized = self.norm1(sequence)
        mask = None
        if temporal_padding_mask is not None:
            if temporal_padding_mask.shape != (batch_size, time_steps):
                raise ValueError("temporal_padding_mask must have shape [B, T].")
            mask = temporal_padding_mask[:, None, :].expand(batch_size, num_tokens, time_steps).reshape(batch_size * num_tokens, time_steps)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=mask,
            need_weights=False,
        )
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.norm2(sequence))
        return sequence.view(batch_size, num_tokens, time_steps, dim).permute(0, 2, 1, 3)


class TemporalFusionTransformer(nn.Module):
    """Explicit sequence fusion after per-frame UniverSat feature extraction.

    Input tokens are expected to preserve the temporal axis: `[B, T, N, input_dim]`.
    The module performs factorized temporal attention at each spatial position, then
    reads a directional change token per position and a small set of global event tokens.
    """

    def __init__(self, config: TemporalFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalFusionConfig()
        cfg = self.config
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if cfg.depth < 1:
            raise ValueError("depth must be at least one.")
        if cfg.event_queries < 1:
            raise ValueError("event_queries must be at least one.")

        self.input_projection = nn.Linear(cfg.input_dim, cfg.model_dim)
        self.time_index_embedding = nn.Embedding(cfg.max_time_steps, cfg.model_dim)
        self.timestamp_projection = nn.Sequential(
            nn.Linear(1, cfg.model_dim),
            nn.Tanh(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalAxisBlock(cfg.model_dim, cfg.num_heads, cfg.ffn_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        )

        self.change_query = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)
        self.change_attention = nn.MultiheadAttention(
            cfg.model_dim,
            cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.change_norm = nn.LayerNorm(cfg.model_dim)

        self.event_queries = nn.Parameter(torch.randn(cfg.event_queries, cfg.model_dim) * 0.02)
        event_layer = nn.TransformerDecoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.model_dim * cfg.ffn_ratio,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.event_decoder = nn.TransformerDecoder(event_layer, num_layers=cfg.event_decoder_layers)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(cfg.model_dim * 2),
            nn.Linear(cfg.model_dim * 2, cfg.output_dim),
        )

    def _time_embeddings(self, batch_size: int, time_steps: int, device: torch.device, dtype: torch.dtype, timestamps: Tensor | None) -> Tensor:
        if time_steps > self.config.max_time_steps:
            raise ValueError(f"T={time_steps} exceeds max_time_steps={self.config.max_time_steps}.")
        indices = torch.arange(time_steps, device=device)
        embedding = self.time_index_embedding(indices).to(dtype).view(1, time_steps, 1, -1)
        embedding = embedding.expand(batch_size, -1, -1, -1)
        if timestamps is None:
            return embedding
        if timestamps.shape != (batch_size, time_steps):
            raise ValueError("timestamps must have shape [B, T].")
        relative = timestamps.to(device=device, dtype=dtype)
        relative = relative - relative[:, :1]
        scale = relative.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        relative = (relative / scale).unsqueeze(-1)
        continuous = self.timestamp_projection(relative).unsqueeze(2)
        return embedding + continuous

    def forward(
        self,
        per_time_tokens: Tensor,
        *,
        timestamps: Tensor | None = None,
        temporal_padding_mask: Tensor | None = None,
    ) -> TemporalFusionOutput:
        if per_time_tokens.ndim != 4:
            raise ValueError("per_time_tokens must have shape [B, T, N, D].")
        batch_size, time_steps, num_tokens, input_dim = per_time_tokens.shape
        if time_steps < 2:
            raise ValueError("TemporalFusionTransformer requires at least two timestamps.")
        if input_dim != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {input_dim}.")

        tokens = self.input_projection(per_time_tokens)
        tokens = tokens + self._time_embeddings(
            batch_size,
            time_steps,
            tokens.device,
            tokens.dtype,
            timestamps,
        )
        for block in self.temporal_blocks:
            tokens = block(tokens, temporal_padding_mask=temporal_padding_mask)

        memory = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, time_steps, self.config.model_dim)
        query = self.change_query.expand(batch_size * num_tokens, -1, -1)
        mask = None
        if temporal_padding_mask is not None:
            mask = temporal_padding_mask[:, None, :].expand(batch_size, num_tokens, time_steps).reshape(batch_size * num_tokens, time_steps)
        change, attention = self.change_attention(
            query,
            memory,
            memory,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )
        change_tokens = self.change_norm(change.squeeze(1)).view(batch_size, num_tokens, self.config.model_dim)
        temporal_attention = attention.squeeze(1).view(batch_size, num_tokens, time_steps)

        event_queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        event_tokens = self.event_decoder(event_queries, change_tokens)
        pooled_change = change_tokens.mean(dim=1)
        pooled_events = event_tokens.mean(dim=1)
        global_embedding = F.normalize(
            self.output_projection(torch.cat([pooled_change, pooled_events], dim=-1)),
            dim=-1,
        )
        return TemporalFusionOutput(
            temporal_tokens=tokens,
            change_tokens=change_tokens,
            event_tokens=event_tokens,
            global_embedding=global_embedding,
            temporal_attention=temporal_attention,
        )
