from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalChangeEncoderConfig:
    input_dim: int = 768
    hidden_dim: int = 512
    depth: int = 4
    heads: int = 8
    ffn_dim: int = 2048
    grid_size: int = 32
    window_size: int = 8
    global_tokens: int = 4
    dropout: float = 0.0
    drop_path_max: float = 0.1
    use_direction_embeddings: bool = False
    use_explicit_change_fusion: bool = False
    max_time_steps: int = 2


@dataclass(frozen=True)
class TemporalChangeEncoderOutput:
    per_time_tokens: Tensor
    change_tokens: Tensor
    global_tokens: Tensor
    pair_embedding: Tensor


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability <= 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def _window_partition(x: Tensor, grid_size: int, window_size: int, shift: bool) -> Tensor:
    batch_size, tokens, dim = x.shape
    if tokens != grid_size * grid_size:
        raise ValueError(f"Expected {grid_size * grid_size} tokens, got {tokens}.")
    grid = x.reshape(batch_size, grid_size, grid_size, dim)
    if shift:
        offset = window_size // 2
        grid = torch.roll(grid, shifts=(-offset, -offset), dims=(1, 2))
    windows = grid.reshape(batch_size, grid_size // window_size, window_size, grid_size // window_size, window_size, dim)
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, dim)
    return windows


def _window_reverse(windows: Tensor, batch_size: int, grid_size: int, window_size: int, shift: bool) -> Tensor:
    dim = windows.shape[-1]
    grid = windows.reshape(batch_size, grid_size // window_size, grid_size // window_size, window_size, window_size, dim)
    grid = grid.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, grid_size, grid_size, dim)
    if shift:
        offset = window_size // 2
        grid = torch.roll(grid, shifts=(offset, offset), dims=(1, 2))
    return grid.reshape(batch_size, grid_size * grid_size, dim)


class TemporalChangeBlock(nn.Module):
    def __init__(self, config: TemporalChangeEncoderConfig, drop_path: float, shift_windows: bool):
        super().__init__()
        self.config = config
        self.shift_windows = shift_windows
        self.temporal_norm = nn.LayerNorm(config.hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(config.hidden_dim, config.heads, dropout=config.dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(config.hidden_dim)
        self.spatial_attn = nn.MultiheadAttention(config.hidden_dim, config.heads, dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, temporal_valid_mask: Tensor | None = None) -> Tensor:
        batch_size, time_steps, spatial_tokens, hidden_dim = x.shape
        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(batch_size * spatial_tokens, time_steps, hidden_dim)
        if temporal_valid_mask is not None:
            key_padding_mask = ~temporal_valid_mask.to(torch.bool).repeat_interleave(spatial_tokens, dim=0)
        else:
            key_padding_mask = None
        temporal_out, _ = self.temporal_attn(temporal, temporal, temporal, key_padding_mask=key_padding_mask, need_weights=False)
        temporal_out = temporal_out.reshape(batch_size, spatial_tokens, time_steps, hidden_dim).permute(0, 2, 1, 3)
        x = x + self.drop_path(temporal_out)

        fused = x.mean(dim=1)
        windows = _window_partition(
            self.spatial_norm(fused),
            self.config.grid_size,
            self.config.window_size,
            self.shift_windows,
        )
        spatial_out, _ = self.spatial_attn(windows, windows, windows, need_weights=False)
        spatial_out = _window_reverse(spatial_out, batch_size, self.config.grid_size, self.config.window_size, self.shift_windows)
        x = x + self.drop_path(spatial_out.unsqueeze(1))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class TemporalChangeEncoder(nn.Module):
    def __init__(self, config: TemporalChangeEncoderConfig | None = None):
        super().__init__()
        self.config = config or TemporalChangeEncoderConfig()
        if self.config.grid_size % self.config.window_size != 0:
            raise ValueError("grid_size must be divisible by window_size")
        if self.config.max_time_steps < 2:
            raise ValueError("max_time_steps must be at least two")
        self.input_projection = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        if self.config.use_direction_embeddings:
            self.direction_embeddings: nn.Parameter | None = nn.Parameter(
                torch.randn(self.config.max_time_steps, self.config.hidden_dim) * 0.02
            )
        else:
            self.direction_embeddings = None
        if self.config.use_explicit_change_fusion:
            fusion_dim = 5 * self.config.hidden_dim
            self.change_fusion: nn.Module | None = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
        else:
            self.change_fusion = None
        drop_rates = torch.linspace(0.0, self.config.drop_path_max, self.config.depth).tolist()
        self.blocks = nn.ModuleList(
            TemporalChangeBlock(self.config, drop_path=float(drop_rates[index]), shift_windows=bool(index % 2))
            for index in range(self.config.depth)
        )
        self.global_queries = nn.Parameter(torch.randn(self.config.global_tokens, self.config.hidden_dim) * 0.02)
        self.global_cross_attn = nn.MultiheadAttention(self.config.hidden_dim, self.config.heads, dropout=self.config.dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)

    def _inject_change_inductive_bias(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        _, time_steps, _, _ = x.shape
        if self.direction_embeddings is not None:
            if time_steps > self.direction_embeddings.shape[0]:
                raise ValueError(
                    f"Received {time_steps} timestamps but only {self.direction_embeddings.shape[0]} direction embeddings are configured"
                )
            x = x + self.direction_embeddings[:time_steps].view(1, time_steps, 1, -1)

        explicit_change: Tensor | None = None
        if self.change_fusion is not None:
            if time_steps != 2:
                raise ValueError("Explicit change fusion currently requires exactly two timestamps")
            before = x[:, 0]
            after = x[:, 1]
            fusion_input = torch.cat(
                [before, after, after - before, (after - before).abs(), before * after],
                dim=-1,
            )
            explicit_change = self.change_fusion(fusion_input)
            x = x + explicit_change.unsqueeze(1)
        return x, explicit_change

    def forward(self, features: Tensor, temporal_valid_mask: Tensor | None = None) -> TemporalChangeEncoderOutput:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B,T,N,D], got {tuple(features.shape)}")
        batch_size, time_steps, spatial_tokens, _ = features.shape
        if temporal_valid_mask is not None and tuple(temporal_valid_mask.shape) != (batch_size, time_steps):
            raise ValueError(
                f"temporal_valid_mask must have shape {(batch_size, time_steps)}, got {tuple(temporal_valid_mask.shape)}"
            )
        if (self.config.use_direction_embeddings or self.config.use_explicit_change_fusion) and time_steps != 2:
            raise ValueError("Direction-aware Stage-1 temporal encoding requires exactly two timestamps")
        if spatial_tokens != self.config.grid_size * self.config.grid_size:
            raise ValueError(f"Expected {self.config.grid_size * self.config.grid_size} spatial tokens, got {spatial_tokens}.")
        x = self.input_projection(features)
        x, explicit_change = self._inject_change_inductive_bias(x)
        for block in self.blocks:
            x = block(x, temporal_valid_mask=temporal_valid_mask)
        if temporal_valid_mask is None:
            change_tokens = x.mean(dim=1)
        else:
            weights = temporal_valid_mask.to(dtype=x.dtype).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
            change_tokens = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if explicit_change is not None:
            change_tokens = change_tokens + explicit_change
        change_tokens = self.output_norm(change_tokens)
        queries = self.global_queries.unsqueeze(0).expand(batch_size, -1, -1)
        global_tokens, _ = self.global_cross_attn(queries, change_tokens, change_tokens, need_weights=False)
        pair_embedding = F.normalize(global_tokens.mean(dim=1), dim=-1)
        return TemporalChangeEncoderOutput(
            per_time_tokens=x,
            change_tokens=change_tokens,
            global_tokens=global_tokens,
            pair_embedding=pair_embedding,
        )
