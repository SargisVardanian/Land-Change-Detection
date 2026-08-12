from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TemporalSigLIPConfig:
    """Typed configuration for the direct TemporalSigLIP model."""

    hidden_size: int = 768
    patch_tokens: int = 256
    patch_grid: int = 16
    max_frames: int = 8
    temporal_layers: int = 2
    attention_heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.0
    max_logit_scale: float = 4.0
    min_logit_scale: float = -5.0
    initial_temperature: float = 0.07
    localization_temperature: float = 0.07

    def validate(self) -> "TemporalSigLIPConfig":
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.patch_grid * self.patch_grid != self.patch_tokens:
            raise ValueError("patch_grid must square to patch_tokens")
        if self.max_frames < 2:
            raise ValueError("max_frames must be at least two")
        if self.temporal_layers != 2:
            raise ValueError("TemporalSigLIP requires exactly two temporal blocks")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.mlp_ratio != 4:
            raise ValueError("the initial model requires an FFN ratio of four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if self.localization_temperature <= 0.0:
            raise ValueError("localization_temperature must be positive")
        if self.min_logit_scale >= self.max_logit_scale:
            raise ValueError("logit-scale bounds are invalid")
        return self

    @property
    def initial_logit_scale(self) -> float:
        return math.log(1.0 / self.initial_temperature)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalSigLIPConfig":
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown TemporalSigLIP config fields: {unknown}")
        return cls(**value).validate()
