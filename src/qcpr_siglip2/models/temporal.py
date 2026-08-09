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
        timestamps: Tensor | None = None,
    ) -> Tensor:
        b, t, n, d = patches.shape
        query = self.query(patches)
        outputs: list[Tensor] = []
        scale = float(d) ** -0.5
        if timestamps is not None and timestamps.shape != (b, t):
            raise ValueError("timestamps must be [B,T] for local temporal attention")
        if timestamps is not None:
            timestamp_values = timestamps.to(device=patches.device, dtype=patches.dtype)
            timestamp_scale = (
                (timestamp_values - timestamp_values[:, :1])
                .abs()
                .amax(dim=1, keepdim=True)
                .clamp_min(1.0)
            )
        else:
            timestamp_values = None
            timestamp_scale = None
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
            if timestamp_values is not None and timestamp_scale is not None:
                source_delta = (
                    (
                        timestamp_values[:, frame_offsets]
                        - timestamp_values[:, target_frame].unsqueeze(-1)
                    )
                    / timestamp_scale
                )
                logits = logits + source_delta.unsqueeze(1)
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
        self.spatial_relative_scale = nn.Parameter(
            torch.zeros(config.attention_heads)
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
        spatial_coordinates: Tensor,
        timestamps: Tensor | None = None,
    ) -> Tensor:
        batch = hidden.shape[0]
        special_count = 1 + frame_count + change_token_count
        special = hidden[:, :special_count]
        patches = hidden[:, special_count:].reshape(
            batch, frame_count, patch_count, self.hidden_size
        )
        flat_patches = patches.reshape(batch * frame_count, patch_count, self.hidden_size)
        flat_valid = patch_valid_mask.reshape(batch * frame_count, patch_count)
        flat_coordinates = spatial_coordinates.reshape(batch * frame_count, patch_count, 2)
        relative_distance = (
            flat_coordinates[:, :, None, :] - flat_coordinates[:, None, :, :]
        ).abs().sum(dim=-1)
        relative_bias = (
            -relative_distance[:, None, :, :]
            * self.spatial_relative_scale.to(dtype=relative_distance.dtype)[None, :, None, None]
        ).reshape(batch * frame_count * self.spatial_attention.num_heads, patch_count, patch_count)
        spatial_key_padding = torch.zeros(
            flat_valid.shape, device=flat_valid.device, dtype=relative_bias.dtype
        ).masked_fill(~flat_valid, -1e4)
        spatial, _ = self.spatial_attention(
            self.spatial_norm(flat_patches),
            self.spatial_norm(flat_patches),
            self.spatial_norm(flat_patches),
            key_padding_mask=spatial_key_padding,
            attn_mask=relative_bias,
            need_weights=False,
        )
        flat_patches = flat_patches + spatial * self.attn_scale
        patches = flat_patches.reshape(batch, frame_count, patch_count, self.hidden_size)
        patches = patches + self.temporal_attention(
            patches, patch_valid_mask, spatial_shapes, timestamps
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
        self.temporal_fuse = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.fusion_scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_init)
        )
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        with torch.no_grad():
            self.temporal_fuse.weight.zero_()
            self.temporal_fuse.weight[:, : config.hidden_size].copy_(
                torch.eye(config.hidden_size)
            )

    def _attend(self, tokens: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("region reducer expects [B,N,D] tokens and [B,N] mask")
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("every large-scene frame needs a valid patch")
        query = self.latent_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        reduced, weights = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return query + reduced * self.scale, weights

    def forward(
        self,
        tokens: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if tokens.ndim != 4 or valid_mask.shape != tokens.shape[:3]:
            raise ValueError(
                "region reducer expects [B,T,N,D] tokens and [B,T,N] mask"
            )
        batch, frames, patches, _hidden = tokens.shape
        if frames < 2:
            raise ValueError("large-scene reduction requires at least two frames")
        valid_float = valid_mask.to(dtype=tokens.dtype)
        denominator = valid_float.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        mean_tokens = (tokens * valid_float.unsqueeze(-1)).sum(dim=1) / denominator
        delta_tokens = tokens[:, -1] - tokens[:, 0]
        fused_tokens = self.temporal_fuse(
            torch.cat((mean_tokens, delta_tokens), dim=-1)
        )
        fused_mask = valid_mask.all(dim=1) | valid_mask.any(dim=1)
        fused_reduced, fused_assignment = self._attend(fused_tokens, fused_mask)

        frame_reduced: list[Tensor] = []
        for frame_index in range(frames):
            reduced, _ = self._attend(tokens[:, frame_index], valid_mask[:, frame_index])
            frame_reduced.append(reduced)
        reduced = torch.stack(frame_reduced, dim=1)
        reduced = reduced + fused_reduced.unsqueeze(1) * self.fusion_scale
        # The fused assignment is shared across frames so a reduced region has
        # one stable native spatial footprint for the whole temporal item.
        assignment = fused_assignment.unsqueeze(1).expand(
            batch, frames, fused_assignment.shape[1], patches
        )
        return reduced, assignment, fused_reduced


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
    native_patch_valid_mask: Tensor | None = None
    native_spatial_shapes: Tensor | None = None
    native_token_coordinates: Tensor | None = None
    region_assignment: Tensor | None = None
    native_image_size: Tensor | None = None
    processed_patch_grid: Tensor | None = None
    transform_hash: str | None = None
    frame_ids: Tensor | None = None
    sensor_ids: Tensor | None = None
    gsd: Tensor | None = None
    metadata_missing: Tensor | None = None
    timestamps: Tensor | None = None
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
        self.time_projection = nn.Linear(
            4 * config.temporal_fourier_bands, config.hidden_size, bias=False
        )
        nn.init.zeros_(self.time_projection.weight)
        self.frame_id_projection = nn.Linear(1, config.hidden_size, bias=False)
        self.gsd_projection = nn.Linear(1, config.hidden_size, bias=False)
        self.sensor_embedding = nn.Embedding(
            config.sensor_vocab_size, config.hidden_size
        )
        nn.init.zeros_(self.frame_id_projection.weight)
        nn.init.zeros_(self.gsd_projection.weight)
        nn.init.zeros_(self.sensor_embedding.weight)
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
        self.region_geometry_projection = nn.Linear(
            4, config.hidden_size, bias=False
        )
        nn.init.zeros_(self.region_geometry_projection.weight)
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

    def _coordinate_position(self, coordinates: Tensor) -> Tensor:
        """Sample the learned 2-D position table at normalized native coordinates."""

        if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
            raise ValueError("token coordinates must be [B,T,N,2]")
        batch, frames, tokens, _ = coordinates.shape
        table = self.spatial_position.float().expand(batch * frames, -1, -1, -1)
        grid = coordinates.float().reshape(batch * frames, tokens, 1, 2)
        grid = grid.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            table,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, frames, tokens, self.config.hidden_size
        ).to(dtype=coordinates.dtype)

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
        delta_scale = delta.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        absolute_scale = timestamp_values.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        relative = delta / delta_scale
        absolute = timestamp_values / absolute_scale
        bands = torch.arange(
            1,
            self.config.temporal_fourier_bands + 1,
            device=device,
            dtype=dtype,
        )
        relative_angles = relative.unsqueeze(-1) * bands
        absolute_angles = absolute.unsqueeze(-1) * bands
        return torch.cat(
            (
                relative_angles.sin(),
                relative_angles.cos(),
                absolute_angles.sin(),
                absolute_angles.cos(),
            ),
            dim=-1,
        )

    def _metadata_bias(
        self,
        *,
        timestamps: Tensor | None,
        frame_ids: Tensor | None,
        sensor_ids: Tensor | None,
        gsd: Tensor | None,
        metadata_missing: Tensor | None,
        batch: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        missing = torch.ones(
            batch, frames, 4, dtype=torch.bool, device=device
        )
        explicit_missing = metadata_missing is not None
        if metadata_missing is not None:
            if metadata_missing.shape != (batch, frames, 4):
                raise ValueError("metadata_missing must be [B,T,4]")
            missing = metadata_missing.to(device=device, dtype=torch.bool)
        if timestamps is not None and not explicit_missing:
            missing[:, :, 0] = False
        if frame_ids is None:
            frame_values = torch.arange(frames, device=device, dtype=dtype).view(1, frames)
            frame_values = frame_values.expand(batch, -1)
        else:
            if frame_ids.shape != (batch, frames):
                raise ValueError("frame_ids must be [B,T]")
            frame_values = frame_ids.to(device=device, dtype=dtype)
            if not explicit_missing:
                missing[:, :, 1] = False
        if sensor_ids is None:
            sensor_values = torch.zeros(
                batch, frames, dtype=torch.long, device=device
            )
        else:
            if sensor_ids.shape != (batch, frames):
                raise ValueError("sensor_ids must be [B,T]")
            sensor_values = sensor_ids.to(device=device, dtype=torch.long)
            if torch.any(sensor_values < 0) or torch.any(
                sensor_values >= self.config.sensor_vocab_size
            ):
                raise ValueError("sensor_ids exceed sensor_vocab_size")
            if not explicit_missing:
                missing[:, :, 2] = False
        if gsd is None:
            gsd_values = torch.zeros(batch, frames, device=device, dtype=dtype)
        else:
            if gsd.shape not in ((batch, frames), (batch, frames, 1)):
                raise ValueError("gsd must be [B,T] or [B,T,1]")
            gsd_values = gsd.reshape(batch, frames).to(device=device, dtype=dtype)
            if torch.any(gsd_values < 0):
                raise ValueError("gsd must be non-negative")
            if not explicit_missing:
                missing[:, :, 3] = False
        timestamp_bias = torch.zeros(
            batch, frames, self.config.hidden_size, device=device, dtype=dtype
        )
        frame_bias = self.frame_id_projection(frame_values.unsqueeze(-1)).to(dtype=dtype)
        sensor_bias = self.sensor_embedding(sensor_values).to(dtype=dtype)
        gsd_bias = self.gsd_projection(
            torch.log1p(gsd_values).unsqueeze(-1)
        ).to(dtype=dtype)
        frame_bias = frame_bias.masked_fill(missing[:, :, 1:2], 0.0)
        sensor_bias = sensor_bias.masked_fill(missing[:, :, 2:3], 0.0)
        gsd_bias = gsd_bias.masked_fill(missing[:, :, 3:4], 0.0)
        return timestamp_bias, frame_bias, sensor_bias, gsd_bias, missing

    def _reduce_large_scene(
        self,
        frame_tokens: Tensor,
        valid_mask: Tensor,
        native_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, t, _n, _ = frame_tokens.shape
        k = self.config.large_scene_latents
        positioned_tokens = frame_tokens + self._coordinate_position(native_coordinates)
        reduced, assignment, _ = self.region_reducer(positioned_tokens, valid_mask)
        reduced_mask = torch.ones(
            b, t, k, dtype=torch.bool, device=frame_tokens.device
        )
        reduced_shapes = torch.tensor(
            [1, k], dtype=torch.long, device=frame_tokens.device
        ).view(1, 1, 2).expand(b, t, 2)
        region_coordinates = torch.einsum(
            "btkn,btnd->btkd", assignment, native_coordinates
        )
        coordinate_second_moment = torch.einsum(
            "btkn,btnd->btkd", assignment, native_coordinates.square()
        )
        region_scale = (
            coordinate_second_moment - region_coordinates.square()
        ).clamp_min(0.0).sqrt()
        geometry = torch.cat((region_coordinates, region_scale), dim=-1)
        reduced = reduced + self.region_geometry_projection(geometry.float()).to(
            dtype=reduced.dtype
        )
        return reduced, reduced_mask, reduced_shapes, assignment, region_coordinates

    def forward(
        self,
        frame_tokens: Tensor,
        frame_embeddings: Tensor,
        timestamps: Tensor | None = None,
        *,
        patch_valid_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        native_image_size: Tensor | None = None,
        processed_patch_grid: Tensor | None = None,
        transform_hash: str | None = None,
        frame_ids: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        gsd: Tensor | None = None,
        metadata_missing: Tensor | None = None,
        token_coordinates: Tensor | None = None,
        force_region_reduction: bool = False,
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
        if not torch.all((shapes == shapes[:, :1]).all(dim=-1)):
            raise ValueError("all frames of a temporal item need a compatible patch grid")
        if native_image_size is not None:
            if native_image_size.shape != (b, t, 2):
                raise ValueError("native_image_size must be [B,T,2]")
            native_image_size = native_image_size.to(
                device=frame_tokens.device, dtype=torch.long
            )
        if processed_patch_grid is not None:
            if processed_patch_grid.shape != (b, t, 2):
                raise ValueError("processed_patch_grid must be [B,T,2]")
            processed_patch_grid = processed_patch_grid.to(
                device=frame_tokens.device, dtype=torch.long
            )
            if not torch.equal(processed_patch_grid, shapes):
                raise ValueError(
                    "processed_patch_grid must match spatial_shapes"
                )
        if token_coordinates is not None:
            if token_coordinates.shape != (b, t, n, 2):
                raise ValueError("token_coordinates must be [B,T,N,2]")
            native_coordinates = token_coordinates.to(
                device=frame_tokens.device, dtype=frame_tokens.dtype
            )
        else:
            _, native_coordinates = self._spatial_metadata(
                valid_mask, shapes, dtype=frame_tokens.dtype
            )
        native_valid_mask = valid_mask
        native_shapes = shapes
        pair_initial = F.normalize(frame_embeddings.sum(dim=1), dim=-1)
        reduced = force_region_reduction or n > self.config.direct_patch_token_budget
        region_assignment: Tensor | None = None
        if reduced:
            tokens, valid_mask, shapes, region_assignment, region_coordinates = (
                self._reduce_large_scene(frame_tokens, valid_mask, native_coordinates)
            )
            coordinates = region_coordinates
        else:
            tokens = frame_tokens
            coordinates = native_coordinates
        _, _, token_count, _ = tokens.shape
        if token_coordinates is not None:
            spatial = self._coordinate_position(coordinates)
        else:
            spatial, _ = self._spatial_metadata(
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
        _, frame_id_bias, sensor_bias, gsd_bias, normalized_missing = self._metadata_bias(
            timestamps=timestamps,
            frame_ids=frame_ids,
            sensor_ids=sensor_ids,
            gsd=gsd,
            metadata_missing=metadata_missing,
            batch=b,
            frames=t,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        time_bias = time_bias.masked_fill(
            normalized_missing[:, :, 0].unsqueeze(-1), 0.0
        )
        frame_metadata_bias = frame_id_bias + sensor_bias + gsd_bias
        frame_bias = self.frame_position[:t].to(dtype=tokens.dtype).view(1, t, 1, d)
        token_values = (
            tokens
            + spatial
            + self.patch_type.to(dtype=tokens.dtype).view(1, 1, 1, d)
            + frame_bias
            + time_bias.to(dtype=tokens.dtype).unsqueeze(2)
            + frame_metadata_bias.unsqueeze(2)
        )
        token_values = token_values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        frame_tokens_with_metadata = (
            self.frame_cls_tokens[:t].to(dtype=tokens.dtype).view(1, t, d)
            + frame_bias.squeeze(2)
            + time_bias.to(dtype=tokens.dtype)
            + frame_metadata_bias
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
                        spatial_coordinates=coordinates,
                        timestamps=timestamps,
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
                    spatial_coordinates=coordinates,
                    timestamps=timestamps,
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
                "native_image_size": native_image_size,
                "processed_patch_grid": shapes if processed_patch_grid is None else processed_patch_grid,
                "transform_hash": transform_hash,
                "frame_ids": frame_ids,
                "sensor_ids": sensor_ids,
                "gsd": gsd,
                "metadata_missing": normalized_missing,
                "timestamps": timestamps,
                "region_assignment": region_assignment,
                "reduced": reduced,
            },
            native_patch_valid_mask=native_valid_mask,
            native_spatial_shapes=native_shapes,
            native_token_coordinates=native_coordinates,
            region_assignment=region_assignment,
            native_image_size=native_image_size,
            processed_patch_grid=shapes if processed_patch_grid is None else processed_patch_grid,
            transform_hash=transform_hash,
            frame_ids=frame_ids,
            sensor_ids=sensor_ids,
            gsd=gsd,
            metadata_missing=normalized_missing,
            timestamps=timestamps,
            reduced=reduced,
        )
