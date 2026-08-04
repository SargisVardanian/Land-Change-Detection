"""Spatial and continuous temporal metadata encoding."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def fourier_features(values: Tensor, bands: int) -> Tensor:
    values = values.unsqueeze(-1)
    frequencies = 2.0 ** torch.arange(bands, device=values.device, dtype=values.dtype)
    angles = values * frequencies * math.pi
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SpatialTemporalEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        spatial_bands: int = 4,
        temporal_bands: int = 4,
        frame_vocab: int = 32,
        sensor_vocab: int = 16,
    ):
        super().__init__()
        self.spatial_bands = spatial_bands
        self.temporal_bands = temporal_bands
        self.frame_embedding = nn.Embedding(frame_vocab, hidden_dim)
        self.sensor_embedding = nn.Embedding(sensor_vocab, hidden_dim)
        feature_dim = 4 * spatial_bands + 4 * temporal_bands + 2
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        coordinates: Tensor,
        timestamps: Tensor,
        frame_ids: Tensor,
        *,
        delta_times: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        gsd: Tensor | None = None,
        metadata_missing: Tensor | None = None,
    ) -> Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must be [B,N,2]")
        if timestamps.ndim != 2 or frame_ids.shape != timestamps.shape:
            raise ValueError("timestamps and frame_ids must be [B,T]")
        batch, n = coordinates.shape[:2]
        time_count = timestamps.shape[1]
        spatial = fourier_features(coordinates, self.spatial_bands).reshape(batch, n, -1).unsqueeze(1)
        spatial = spatial.expand(-1, time_count, -1, -1)
        time = fourier_features(timestamps, self.temporal_bands).unsqueeze(2).expand(-1, -1, n, -1)
        delta = timestamps - timestamps[:, :1] if delta_times is None else delta_times
        delta_features = fourier_features(delta, self.temporal_bands).unsqueeze(2).expand(-1, -1, n, -1)
        if gsd is None:
            gsd_feature = torch.zeros((batch, time_count, n, 1), device=timestamps.device, dtype=timestamps.dtype)
        else:
            if gsd.ndim == 2:
                gsd = gsd.unsqueeze(-1)
            gsd_feature = torch.log1p(gsd.clamp_min(0.0)).unsqueeze(2).expand(-1, -1, n, -1)
        if metadata_missing is None:
            missing = torch.zeros((batch, time_count, 1), device=timestamps.device, dtype=timestamps.dtype)
        else:
            missing = metadata_missing.to(timestamps.dtype)
            if missing.ndim == 2:
                missing = missing.unsqueeze(-1)
        missing = missing.unsqueeze(2).expand(-1, -1, n, -1)
        raw = torch.cat([spatial, time, delta_features, gsd_feature, missing], dim=-1)
        if raw.shape[-1] != self.project[0].in_features:
            raise ValueError("metadata feature width does not match positional encoder")
        frame = self.frame_embedding(frame_ids.clamp_min(0)).unsqueeze(2)
        sensor = torch.zeros_like(frame)
        if sensor_ids is not None:
            sensor = self.sensor_embedding(sensor_ids.clamp_min(0)).unsqueeze(2)
        return self.project(raw) + frame + sensor
