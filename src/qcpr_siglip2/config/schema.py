from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Siglip2TemporalConfig:
    backbone: str = "google/siglip2-base-patch16-naflex"
    hidden_size: int = 768
    # ``None`` is intentional.  The native NaFlex sequence length is a
    # runtime property of the image and of ``max_num_patches``; it must not be
    # silently reduced to the historical 16x16 contract.
    expected_patch_tokens: int | None = None
    # This is the interpolation base for the learned 2-D position table, not
    # a restriction on the native image grid.  An integer remains accepted for
    # backwards-compatible JSON/configuration files.
    base_grid: int | tuple[int, int] = (16, 16)
    temporal_layers: int = 2
    attention_heads: int = 12
    mlp_size: int = 3072
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    layer_scale_init: float = 1e-4
    max_frames: int = 8
    change_token_count: int = 4
    local_displacement_radius: int = 1
    temporal_fourier_bands: int = 4
    sensor_vocab_size: int = 64
    evidence_temperature: float = 0.07
    retrieval_temperature: float = 0.07
    evidence_gate_init: float = 0.05
    evidence_query_chunk_size: int = 16
    evidence_pair_chunk_size: int = 8
    text_pad_id: int | None = None
    patch_size: int = 16
    supported_patch_budgets: tuple[int, ...] = (256, 576, 1024)
    direct_patch_token_budget: int = 1024
    large_scene_latents: int = 64
    retrieval_score_mode: Literal[
        "final_v1_primary", "evidence_mechanism_ablation"
    ] = "final_v1_primary"

    def validate(self) -> Siglip2TemporalConfig:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.expected_patch_tokens is not None and self.expected_patch_tokens <= 0:
            raise ValueError("expected_patch_tokens must be positive when supplied")
        grid = self.base_grid
        if isinstance(grid, int):
            grid = (grid, grid)
        if (
            not isinstance(grid, tuple)
            or len(grid) != 2
            or any(not isinstance(side, int) or side <= 0 for side in grid)
        ):
            raise ValueError("base_grid must be a positive integer or (height, width)")
        if self.temporal_layers != 2:
            raise ValueError("the minimal track requires exactly two temporal layers")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.mlp_size != 4 * self.hidden_size:
            raise ValueError("the minimal track requires an FFN ratio of four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean")
        if self.layer_scale_init <= 0.0 or self.layer_scale_init > 1.0:
            raise ValueError("layer_scale_init must be in (0, 1]")
        if self.max_frames < 2:
            raise ValueError("at least two frames are required")
        if self.change_token_count <= 0:
            raise ValueError("change_token_count must be positive")
        if self.local_displacement_radius < 0:
            raise ValueError("local_displacement_radius must be non-negative")
        if self.temporal_fourier_bands <= 0:
            raise ValueError("temporal_fourier_bands must be positive")
        if self.sensor_vocab_size <= 0:
            raise ValueError("sensor_vocab_size must be positive")
        if self.evidence_temperature <= 0.0 or self.retrieval_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if not 0.0 < self.evidence_gate_init < 0.5:
            raise ValueError("evidence_gate_init must lie in (0, 0.5)")
        if self.evidence_query_chunk_size <= 0 or self.evidence_pair_chunk_size <= 0:
            raise ValueError("evidence chunk sizes must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not self.supported_patch_budgets or any(
            not isinstance(budget, int) or budget <= 0
            for budget in self.supported_patch_budgets
        ):
            raise ValueError("supported_patch_budgets must contain positive integers")
        if self.direct_patch_token_budget <= 0:
            raise ValueError("direct_patch_token_budget must be positive")
        if self.large_scene_latents <= 0:
            raise ValueError("large_scene_latents must be positive")
        if self.retrieval_score_mode not in {
            "final_v1_primary",
            "evidence_mechanism_ablation",
        }:
            raise ValueError("unsupported retrieval_score_mode")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Siglip2TemporalConfig:
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown SigLIP-2 config fields: {unknown}")
        normalized = dict(value)
        if isinstance(normalized.get("base_grid"), list):
            normalized["base_grid"] = tuple(normalized["base_grid"])
        if isinstance(normalized.get("supported_patch_budgets"), list):
            normalized["supported_patch_budgets"] = tuple(
                normalized["supported_patch_budgets"]
            )
        return cls(**normalized).validate()

    @property
    def evidence_gate_raw_init(self) -> float:
        return math.log(self.evidence_gate_init / (0.5 - self.evidence_gate_init))
