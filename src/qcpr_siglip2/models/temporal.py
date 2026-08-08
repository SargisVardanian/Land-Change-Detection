from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..config.schema import Siglip2TemporalConfig
from ..contracts import normalize_patch_metadata


class LocalCrossTimeAttention(nn.Module):
    """Linear-in-token local temporal attention with displacement candidates."""

    def __init__(self, config: Siglip2TemporalConfig, radius: int = 1) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("local temporal radius must be non-negative")
        self.radius = radius
        self.offsets = tuple(
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        )
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.displacement_bias = nn.Parameter(torch.zeros(len(self.offsets)))
        self.relative_frame_bias = nn.Parameter(torch.zeros(config.max_frames))

    def forward(
        self,
        patches: Tensor,
        valid_mask: Tensor,
        spatial_shapes: Tensor,
    ) -> Tensor:
        b, t, n, d = patches.shape
        query = self.query(patches)
        outputs: list[Tensor] = []
        scale = float(d) ** -0.5
        token_index = torch.arange(n, device=patches.device).view(1, n)
        for target_frame in range(t):
            target_width = spatial_shapes[:, target_frame, 1].view(b, 1)
            target_y = token_index // target_width
            target_x = token_index.remainder(target_width)
            target_valid = valid_mask[:, target_frame]
            keys: list[Tensor] = []
            values: list[Tensor] = []
            candidate_valid: list[Tensor] = []
            for source_frame in range(t):
                source_height = spatial_shapes[:, source_frame, 0].view(b, 1)
                source_width = spatial_shapes[:, source_frame, 1].view(b, 1)
                source_values = patches[:, source_frame]
                source_keys = self.key(source_values)
                source_projected_values = self.value(source_values)
                for dy, dx in self.offsets:
                    source_y = target_y + dy
                    source_x = target_x + dx
                    inside = (
                        (source_y >= 0)
                        & (source_y < source_height)
                        & (source_x >= 0)
                        & (source_x < source_width)
                    )
                    source_index = (source_y.clamp_min(0) * source_width + source_x.clamp_min(0)).clamp(
                        max=n - 1
                    )
                    gathered_index = source_index.unsqueeze(-1).expand(-1, -1, d)
                    keys.append(torch.gather(source_keys, 1, gathered_index))
                    values.append(
                        torch.gather(source_projected_values, 1, gathered_index)
                    )
                    candidate_valid.append(
                        inside
                        & target_valid
                        & torch.gather(valid_mask[:, source_frame], 1, source_index)
                    )
            key_tensor = torch.stack(keys, dim=2)
            value_tensor = torch.stack(values, dim=2)
            valid_tensor = torch.stack(candidate_valid, dim=2)
            logits = (
                query[:, target_frame].unsqueeze(2) * key_tensor
            ).sum(dim=-1) * scale
            logits = logits + self.displacement_bias.repeat(t)[: logits.shape[2]]
            # Keep temporal order available to the adapter while maintaining a
            # single query-independent pair embedding.
            frame_offsets = torch.arange(t, device=patches.device).repeat_interleave(
                len(self.offsets)
            )
            logits = logits + self.relative_frame_bias[frame_offsets].view(1, 1, -1)
            logits = logits.masked_fill(~valid_tensor, -1e4)
            weights = torch.softmax(logits, dim=2)
            output = (weights.unsqueeze(-1) * value_tensor).sum(dim=2)
            output = self.output(output).masked_fill(~target_valid.unsqueeze(-1), 0.0)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


class LayerScaleTransformerBlock(nn.Module):
    """Structured temporal block: spatial, local-time and bounded global paths."""

    def __init__(self, config: Siglip2TemporalConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.spatial_norm = nn.LayerNorm(config.hidden_size)
        self.spatial_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.temporal_attention = LocalCrossTimeAttention(
            config, radius=config.local_displacement_radius
        )
        self.global_norm = nn.LayerNorm(config.hidden_size)
        self.patch_norm = nn.LayerNorm(config.hidden_size)
        self.global_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_init)
        )
        self.norm_ffn = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_size, config.hidden_size),
        )
        self.ffn_scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_init)
        )

    def forward(
        self,
        hidden: Tensor,
        key_padding_mask: Tensor,
        *,
        frame_count: int,
        patch_count: int,
        change_token_count: int,
        spatial_shapes: Tensor,
        patch_valid_mask: Tensor,
    ) -> Tensor:
        batch = hidden.shape[0]
        special_count = 1 + frame_count + change_token_count
        special = hidden[:, :special_count]
        patches = hidden[:, special_count:].reshape(
            batch, frame_count, patch_count, self.hidden_size
        )
        flat_patches = patches.reshape(batch * frame_count, patch_count, self.hidden_size)
        flat_valid = patch_valid_mask.reshape(batch * frame_count, patch_count)
        spatial, _ = self.spatial_attention(
            self.spatial_norm(flat_patches),
            self.spatial_norm(flat_patches),
            self.spatial_norm(flat_patches),
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        flat_patches = flat_patches + spatial * self.attn_scale
        patches = flat_patches.reshape(batch, frame_count, patch_count, self.hidden_size)
        patches = patches + self.temporal_attention(
            patches, patch_valid_mask, spatial_shapes
        ) * self.attn_scale
        patches = patches.masked_fill(~patch_valid_mask.unsqueeze(-1), 0.0)
        global_values = patches.reshape(batch, frame_count * patch_count, self.hidden_size)
        global_output, _ = self.global_attention(
            self.global_norm(special),
            self.patch_norm(global_values),
            self.patch_norm(global_values),
            key_padding_mask=key_padding_mask[:, special_count:],
            need_weights=False,
        )
        special = special + global_output * self.attn_scale
        hidden = torch.cat((special, patches.reshape(batch, -1, self.hidden_size)), dim=1)
        hidden = hidden + self.ffn(self.norm_ffn(hidden)) * self.ffn_scale
        return hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class BoundedRegionReducer(nn.Module):
    """Query-independent reducer used when native scenes exceed direct budget.

    Learned region queries attend to the valid native patches.  The reducer
    emits bounded region tokens, never ANN vectors; the temporal adapter then
    produces the single sequence embedding from those tokens.
    """

    def __init__(self, config: Siglip2TemporalConfig) -> None:
        super().__init__()
        self.latent_queries = nn.Parameter(
            torch.zeros(config.large_scene_latents, config.hidden_size)
        )
        self.query_norm = nn.LayerNorm(config.hidden_size)
        self.token_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_init)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("region reducer expects [B,N,D] tokens and [B,N] mask")
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("every large-scene frame needs a valid patch")
        query = self.latent_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        reduced, _ = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        return query + reduced * self.scale


@dataclass
class TemporalAdapterOutput:
    pair_cls: Tensor
    frame_cls: Tensor
    change_tokens: Tensor
    temporal_patch_tokens: Tensor
    pair_initial: Tensor
    frame_count: int
    patch_count: int
    patch_valid_mask: Tensor
    spatial_shapes: Tensor
    token_coordinates: Tensor
    temporal_metadata: dict[str, object]
    reduced: bool = False


class TemporalTransformerAdapter(nn.Module):
    """Two-layer temporal adapter over variable-length native patch tokens."""

    def __init__(self, config: Siglip2TemporalConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.frame_position = nn.Parameter(
            torch.zeros(config.max_frames, config.hidden_size)
        )
        self.frame_cls_tokens = nn.Parameter(
            torch.zeros(config.max_frames, config.hidden_size)
        )
        self.change_tokens = nn.Parameter(
            torch.zeros(config.change_token_count, config.hidden_size)
        )
        self.patch_type = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        base_grid = config.base_grid
        if isinstance(base_grid, int):
            base_grid = (base_grid, base_grid)
        self.spatial_position = nn.Parameter(
            torch.zeros(1, config.hidden_size, base_grid[0], base_grid[1])
        )
        self.time_projection = nn.Linear(2, config.hidden_size, bias=False)
        nn.init.zeros_(self.time_projection.weight)
        # A near-zero directional residual prevents the pair representation
        # from being permutation-invariant at initialization while retaining
        # the accepted pooled baseline to numerical precision.
        self.direction_scale = nn.Parameter(
            torch.tensor(config.layer_scale_init * 10.0)
        )
        nn.init.trunc_normal_(self.frame_position, std=0.02)
        nn.init.trunc_normal_(self.frame_cls_tokens, std=0.02)
        nn.init.trunc_normal_(self.change_tokens, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        self.region_reducer = BoundedRegionReducer(config)
        self.blocks = nn.ModuleList(
            [LayerScaleTransformerBlock(config) for _ in range(config.temporal_layers)]
        )

    def _grid_position(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        value = F.interpolate(
            self.spatial_position.float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        position = value.flatten(2).transpose(1, 2).to(device=device, dtype=dtype)
        ys = torch.linspace(
            0.0, 1.0, height, device=device, dtype=dtype
        )
        xs = torch.linspace(
            0.0, 1.0, width, device=device, dtype=dtype
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        return position, coordinates

    def _spatial_metadata(
        self,
        valid_mask: Tensor,
        spatial_shapes: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        b, t, n = valid_mask.shape
        positions = torch.zeros(
            b, t, n, self.config.hidden_size,
            device=valid_mask.device,
            dtype=dtype,
        )
        coordinates = torch.zeros(
            b, t, n, 2, device=valid_mask.device, dtype=dtype
        )
        for batch_index in range(b):
            for frame_index in range(t):
                height, width = (
                    int(spatial_shapes[batch_index, frame_index, 0].item()),
                    int(spatial_shapes[batch_index, frame_index, 1].item()),
                )
                count = min(n, height * width)
                grid_position, grid_coordinates = self._grid_position(
                    height,
                    width,
                    device=valid_mask.device,
                    dtype=dtype,
                )
                positions[batch_index, frame_index, :count] = grid_position[0, :count]
                coordinates[batch_index, frame_index, :count] = grid_coordinates[
                    0, :count
                ]
        return positions, coordinates

    def _time_features(
        self,
        timestamps: Tensor | None,
        *,
        batch: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if timestamps is None:
            timestamp_values = torch.zeros((batch, frames), device=device, dtype=dtype)
        else:
            if timestamps.shape != (batch, frames):
                raise ValueError("timestamps must be [B,T]")
            timestamp_values = timestamps.to(device=device, dtype=dtype)
        first = timestamp_values[:, :1]
        delta = timestamp_values - first
        scale = delta.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return torch.stack((delta, delta / scale), dim=-1)

    def _reduce_large_scene(
        self,
        frame_tokens: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, t, n, _ = frame_tokens.shape
        k = self.config.large_scene_latents
        flat = frame_tokens.reshape(b * t, n, -1)
        flat_mask = valid_mask.reshape(b * t, n)
        reduced = self.region_reducer(flat, flat_mask).reshape(b, t, k, -1)
        reduced_mask = torch.ones(
            b, t, k, dtype=torch.bool, device=frame_tokens.device
        )
        reduced_shapes = torch.tensor(
            [1, k], dtype=torch.long, device=frame_tokens.device
        ).view(1, 1, 2).expand(b, t, 2)
        return reduced, reduced_mask, reduced_shapes

    def forward(
        self,
        frame_tokens: Tensor,
        frame_embeddings: Tensor,
        timestamps: Tensor | None = None,
        *,
        patch_valid_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
    ) -> TemporalAdapterOutput:
        if frame_tokens.ndim != 4 or frame_embeddings.ndim != 3:
            raise ValueError(
                "frame tokens must be [B,T,N,D] and frame embeddings [B,T,D]"
            )
        b, t, n, d = frame_tokens.shape
        if d != self.config.hidden_size or frame_embeddings.shape != (b, t, d):
            raise ValueError("temporal feature shapes do not match config")
        if t < 2 or t > self.config.max_frames:
            raise ValueError("frame count outside supported range")
        valid_mask, shapes = normalize_patch_metadata(
            frame_tokens, patch_valid_mask, spatial_shapes
        )
        pair_initial = F.normalize(frame_embeddings.sum(dim=1), dim=-1)
        reduced = n > self.config.direct_patch_token_budget
        if reduced:
            tokens, valid_mask, shapes = self._reduce_large_scene(
                frame_tokens, valid_mask
            )
        else:
            tokens = frame_tokens
        _, _, token_count, _ = tokens.shape
        spatial, coordinates = self._spatial_metadata(
            valid_mask, shapes, dtype=tokens.dtype
        )
        time_features = self._time_features(
            timestamps,
            batch=b,
            frames=t,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        time_bias = self.time_projection(
            time_features.to(dtype=self.time_projection.weight.dtype)
        ).to(dtype=tokens.dtype)
        frame_bias = self.frame_position[:t].to(dtype=tokens.dtype).view(1, t, 1, d)
        token_values = (
            tokens
            + spatial
            + self.patch_type.to(dtype=tokens.dtype).view(1, 1, 1, d)
            + frame_bias
            + time_bias.to(dtype=tokens.dtype).unsqueeze(2)
        )
        token_values = token_values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        frame_tokens_with_metadata = (
            self.frame_cls_tokens[:t].to(dtype=tokens.dtype).view(1, t, d)
            + frame_bias.squeeze(2)
            + time_bias.to(dtype=tokens.dtype)
        )
        change_tokens = self.change_tokens.to(dtype=tokens.dtype).view(
            1, self.config.change_token_count, d
        )
        hidden = torch.cat(
            [
                pair_initial.unsqueeze(1),
                frame_tokens_with_metadata,
                change_tokens.expand(b, -1, -1),
                token_values.reshape(b, t * token_count, d),
            ],
            dim=1,
        )
        key_padding_mask = torch.cat(
            [
                torch.zeros(
                    b,
                    1 + t + self.config.change_token_count,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                (~valid_mask).reshape(b, t * token_count),
            ],
            dim=1,
        )
        for block in self.blocks:
            if self.training and self.config.gradient_checkpointing:
                hidden = checkpoint(
                    lambda current, mask, module=block: module(
                        current,
                        mask,
                        frame_count=t,
                        patch_count=token_count,
                        change_token_count=self.config.change_token_count,
                        spatial_shapes=shapes,
                        patch_valid_mask=valid_mask,
                    ),
                    hidden,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    key_padding_mask,
                    frame_count=t,
                    patch_count=token_count,
                    change_token_count=self.config.change_token_count,
                    spatial_shapes=shapes,
                    patch_valid_mask=valid_mask,
                )
            hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        directional_residual = (
            frame_embeddings[:, -1] - frame_embeddings[:, 0]
        ).to(dtype=hidden.dtype)
        pair = F.normalize(
            hidden[:, 0] + self.direction_scale.to(dtype=hidden.dtype) * directional_residual,
            dim=-1,
        )
        frame_start = 1
        frame_end = frame_start + t
        change_end = frame_end + self.config.change_token_count
        frame_cls = hidden[:, frame_start:frame_end]
        output_change_tokens = hidden[:, frame_end:change_end]
        patches = hidden[:, change_end:].reshape(b, t * token_count, d)
        return TemporalAdapterOutput(
            pair_cls=pair,
            frame_cls=frame_cls,
            change_tokens=output_change_tokens,
            temporal_patch_tokens=patches,
            pair_initial=pair_initial,
            frame_count=t,
            patch_count=token_count,
            patch_valid_mask=valid_mask,
            spatial_shapes=shapes,
            token_coordinates=coordinates,
            temporal_metadata={
                "native_patch_count": n,
                "processed_patch_count": token_count,
                "patch_valid_mask": valid_mask,
                "spatial_shapes": shapes,
                "token_coordinates": coordinates,
                "reduced": reduced,
            },
            reduced=reduced,
        )
