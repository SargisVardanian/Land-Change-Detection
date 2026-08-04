"""Joint variable-length temporal transformer above native UniverSat tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.contracts import TemporalMetadata
from .positional_encoding import SpatialTemporalEncoding


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass(frozen=True)
class TemporalOutput:
    sequence_cls: Tensor
    frame_cls: Tensor
    change_tokens: Tensor
    dense_tokens: Tensor
    coordinates: Tensor
    frame_mask: Tensor
    token_mask: Tensor
    temporal: TemporalMetadata

    def validate(self) -> None:
        b, t, n, d = self.dense_tokens.shape
        if self.sequence_cls.shape != (b, d):
            raise ValueError("sequence_cls shape is invalid")
        if self.frame_cls.shape != (b, t, d):
            raise ValueError("frame_cls shape is invalid")
        if self.change_tokens.ndim != 3 or self.change_tokens.shape[0] != b or self.change_tokens.shape[2] != d:
            raise ValueError("change_tokens shape is invalid")
        if self.coordinates.shape != (b, n, 2):
            raise ValueError("coordinates shape is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.dense_tokens.shape[0])


class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ff_ratio: int, dropout: float, layer_scale: float, radius: int):
        super().__init__()
        self.radius = radius
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_scale = nn.Parameter(torch.full((hidden_dim,), layer_scale))
        self.spatial_scale = nn.Parameter(torch.full((hidden_dim,), layer_scale))
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.global_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.global_scale = nn.Parameter(torch.full((hidden_dim,), layer_scale))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_ratio, hidden_dim),
        )
        self.ffn_scale = nn.Parameter(torch.full((hidden_dim,), layer_scale))
        offsets = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)
        self.offset_bias = nn.Parameter(torch.zeros(len(offsets)))

    def _local_indices(self, coordinates: Tensor) -> Tensor:
        indices: list[list[int]] = []
        for point_index, point in enumerate(coordinates):
            choices: list[int] = []
            offsets = cast(Tensor, self.offsets)
            for offset in offsets.to(coordinates):
                target = point + offset
                distances = (coordinates - target).abs().sum(dim=-1)
                best = int(distances.argmin())
                choices.append(best if float(distances[best]) < 1e-4 else point_index)
            indices.append(choices)
        return torch.tensor(indices, dtype=torch.long, device=coordinates.device)

    def _local_cross_time(self, x: Tensor, coordinates: Tensor, frame_mask: Tensor, token_mask: Tensor) -> Tensor:
        batch, time_count, n, dim = x.shape
        if time_count < 2 or self.radius == 0:
            return torch.zeros_like(x)
        indices = self._local_indices(coordinates[0])
        width = indices.shape[1]
        outputs: list[Tensor] = []
        for target_frame in range(time_count):
            query = self.cross_norm(x[:, target_frame])
            keys_by_source: list[Tensor] = []
            masks_by_source: list[Tensor] = []
            for source_frame in range(time_count):
                if source_frame == target_frame:
                    continue
                gathered = x[:, source_frame].index_select(1, indices.reshape(-1)).reshape(batch, n, width, dim)
                gathered_mask = token_mask[:, source_frame].index_select(1, indices.reshape(-1)).reshape(batch, n, width)
                source_valid = frame_mask[:, source_frame].view(batch, 1, 1)
                keys_by_source.append(gathered)
                masks_by_source.append(gathered_mask & source_valid)
            keys = torch.cat(keys_by_source, dim=2)
            valid = torch.cat(masks_by_source, dim=2)
            scores = (query.unsqueeze(2) * self.cross_norm(keys)).sum(dim=-1) / (dim ** 0.5)
            scores = scores + self.offset_bias.repeat(time_count - 1).view(1, 1, -1)
            no_valid = ~valid.any(dim=-1)
            safe_valid = valid.clone()
            safe_valid[no_valid, 0] = True
            scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            outputs.append((weights.unsqueeze(-1) * keys).sum(dim=2))
        return torch.stack(outputs, dim=1) * self.cross_scale.view(1, 1, 1, dim)

    def forward(
        self,
        x: Tensor,
        sequence_cls: Tensor,
        frame_cls: Tensor,
        change_tokens: Tensor,
        coordinates: Tensor,
        frame_mask: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, time_count, n, dim = x.shape
        flat = x.reshape(batch * time_count, n, dim)
        flat_mask = ~token_mask.reshape(batch * time_count, n)
        no_valid = ~token_mask.reshape(batch * time_count, n).any(dim=1)
        safe_flat_mask = flat_mask.clone()
        safe_flat_mask[no_valid, 0] = False
        normalized = self.spatial_norm(flat)
        spatial_delta, _ = self.spatial_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=safe_flat_mask,
            need_weights=False,
        )
        x = x + self.spatial_scale.view(1, 1, 1, dim) * spatial_delta.reshape(batch, time_count, n, dim)
        x = x + self._local_cross_time(x, coordinates, frame_mask, token_mask)
        effective_token_mask = token_mask & frame_mask.unsqueeze(-1)
        frame_cls = frame_cls + masked_mean(x, effective_token_mask, dim=2)
        globals_in = torch.cat([sequence_cls.unsqueeze(1), frame_cls, change_tokens], dim=1)
        global_mask = torch.cat(
            [
                torch.ones((batch, 1), dtype=torch.bool, device=x.device),
                frame_mask,
                torch.ones((batch, change_tokens.shape[1]), dtype=torch.bool, device=x.device),
            ],
            dim=1,
        )
        global_norm = self.global_norm(globals_in)
        global_delta, _ = self.global_attn(
            global_norm,
            global_norm,
            global_norm,
            key_padding_mask=~global_mask,
            need_weights=False,
        )
        globals_out = globals_in + self.global_scale.view(1, 1, dim) * global_delta
        sequence_cls = globals_out[:, 0]
        frame_cls = globals_out[:, 1:1 + time_count]
        change_tokens = globals_out[:, 1 + time_count:]
        x = x + sequence_cls[:, None, None, :] * self.global_scale.view(1, 1, 1, dim)
        x = x + self.ffn_scale.view(1, 1, 1, dim) * self.ffn(self.ffn_norm(x))
        return x, sequence_cls, frame_cls, change_tokens


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        native_dim: int = 768,
        hidden_dim: int = 512,
        *,
        heads: int = 8,
        blocks: int = 2,
        change_slots: int = 4,
        ff_ratio: int = 4,
        dropout: float = 0.1,
        layer_scale_init: float = 1e-3,
        local_radius: int = 1,
        spatial_bands: int = 4,
        temporal_bands: int = 4,
        max_frames: int = 16,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Identity() if native_dim == hidden_dim else nn.Linear(native_dim, hidden_dim)
        self.position = SpatialTemporalEncoding(
            hidden_dim,
            spatial_bands=spatial_bands,
            temporal_bands=temporal_bands,
            frame_vocab=max_frames,
        )
        self.change_seed = nn.Parameter(torch.zeros(1, change_slots, hidden_dim))
        self.position_scale = nn.Parameter(torch.tensor(1e-3))
        self.blocks = nn.ModuleList(
            TemporalBlock(hidden_dim, heads, ff_ratio, dropout, layer_scale_init, local_radius)
            for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        native_tokens: Tensor,
        coordinates: Tensor,
        temporal: TemporalMetadata,
        frame_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> TemporalOutput:
        if native_tokens.ndim != 4:
            raise ValueError("native_tokens must be [B,T,N,native_dim]")
        batch, time_count, n, _ = native_tokens.shape
        if coordinates.shape != (batch, n, 2):
            raise ValueError("coordinates must be [B,N,2]")
        if frame_mask is None:
            frame_mask = torch.ones((batch, time_count), dtype=torch.bool, device=native_tokens.device)
        if token_mask is None:
            token_mask = torch.ones((batch, time_count, n), dtype=torch.bool, device=native_tokens.device)
        if frame_mask.shape != (batch, time_count) or token_mask.shape != (batch, time_count, n):
            raise ValueError("temporal masks have invalid shapes")
        x = self.input_projection(native_tokens)
        x = x + self.position_scale * self.position(
            coordinates,
            temporal.timestamps,
            temporal.frame_ids,
            delta_times=temporal.delta_times,
            sensor_ids=temporal.sensor_ids,
            gsd=temporal.gsd,
            metadata_missing=temporal.metadata_missing,
        )
        x = x * token_mask.unsqueeze(-1).to(x.dtype)
        frame_cls = masked_mean(x, token_mask & frame_mask.unsqueeze(-1), dim=2)
        sequence_cls = masked_mean(frame_cls, frame_mask, dim=1)
        change_tokens = self.change_seed.expand(batch, -1, -1)
        for block in self.blocks:
            x, sequence_cls, frame_cls, change_tokens = block(
                x,
                sequence_cls,
                frame_cls,
                change_tokens,
                coordinates,
                frame_mask,
                token_mask,
            )
        x = self.output_norm(x)
        result = TemporalOutput(
            sequence_cls=self.output_norm(sequence_cls),
            frame_cls=self.output_norm(frame_cls),
            change_tokens=self.output_norm(change_tokens),
            dense_tokens=x,
            coordinates=coordinates,
            frame_mask=frame_mask,
            token_mask=token_mask,
            temporal=temporal,
        )
        result.validate()
        return result
