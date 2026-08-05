from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Siglip2TemporalConfig:
    hidden_size: int = 768
    expected_patch_tokens: int = 256
    base_grid: int = 16
    temporal_layers: int = 2
    attention_heads: int = 12
    mlp_size: int = 3072
    dropout: float = 0.0
    layer_scale_init: float = 1e-4
    max_frames: int = 8
    evidence_temperature: float = 0.07
    retrieval_temperature: float = 0.07
    evidence_gate_init: float = 0.05
    evidence_query_chunk_size: int = 16
    evidence_pair_chunk_size: int = 8
    text_pad_id: int | None = None

    def validate(self) -> Siglip2TemporalConfig:
        if self.hidden_size <= 0 or self.expected_patch_tokens <= 0:
            raise ValueError("hidden_size and expected_patch_tokens must be positive")
        if self.base_grid * self.base_grid != self.expected_patch_tokens:
            raise ValueError("base_grid must square to expected_patch_tokens")
        if self.temporal_layers != 2:
            raise ValueError("the minimal track requires exactly two temporal layers")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.mlp_size != 4 * self.hidden_size:
            raise ValueError("the minimal track requires an FFN ratio of four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_scale_init <= 0.0 or self.layer_scale_init > 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")
        if self.max_frames < 2:
            raise ValueError("at least two frames are required")
        if self.evidence_temperature <= 0.0 or self.retrieval_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if not 0.0 < self.evidence_gate_init < 0.5:
            raise ValueError("evidence_gate_init must lie in (0, 0.5)")
        if self.evidence_query_chunk_size <= 0 or self.evidence_pair_chunk_size <= 0:
            raise ValueError("evidence chunk sizes must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Siglip2TemporalConfig:
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown SigLIP-2 config fields: {unknown}")
        return cls(**value).validate()

    @property
    def evidence_gate_raw_init(self) -> float:
        return math.log(self.evidence_gate_init / (0.5 - self.evidence_gate_init))
