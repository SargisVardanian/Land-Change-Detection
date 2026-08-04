"""Typed configuration contracts for QCPR v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a QCPR v3 configuration violates its contract."""


@dataclass(frozen=True)
class TemporalConfig:
    native_dim: int = 768
    hidden_dim: int = 512
    heads: int = 8
    blocks: int = 2
    change_slots: int = 4
    ff_ratio: int = 4
    dropout: float = 0.1
    layer_scale_init: float = 1e-3
    local_radius: int = 1
    spatial_rope_bands: int = 4
    temporal_fourier_bands: int = 4
    max_frames: int = 16

    def validate(self) -> None:
        if self.native_dim <= 0 or self.hidden_dim <= 0:
            raise ConfigError("native_dim and hidden_dim must be positive")
        if self.heads <= 0 or self.hidden_dim % self.heads:
            raise ConfigError("hidden_dim must be divisible by heads")
        if self.blocks < 1 or self.change_slots < 1:
            raise ConfigError("blocks and change_slots must be positive")
        if self.ff_ratio < 1 or self.local_radius < 0:
            raise ConfigError("ff_ratio must be >=1 and local_radius >=0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        if self.layer_scale_init <= 0.0:
            raise ConfigError("layer_scale_init must be positive")
        if self.max_frames < 2:
            raise ConfigError("max_frames must support at least T=2")


@dataclass(frozen=True)
class TextConfig:
    input_dim: int = 1024
    hidden_dim: int = 512
    adapter_layers: int = 2
    heads: int = 8
    ff_ratio: int = 4
    dropout: float = 0.1
    max_length: int = 128
    zero_init_residual: bool = True

    def validate(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ConfigError("text dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ConfigError("text hidden_dim must be divisible by heads")
        if self.adapter_layers < 1 or self.max_length < 1:
            raise ConfigError("text adapter_layers and max_length must be positive")


@dataclass(frozen=True)
class EvidenceConfig:
    enabled: bool = True
    sparse_normalizer: str = "sparsemax"
    temperature: float = 0.07
    top_k: int = 100
    pairwise_query_chunk: int = 16

    def validate(self) -> None:
        if self.sparse_normalizer not in {"sparsemax", "softmax"}:
            raise ConfigError("sparse_normalizer must be sparsemax or softmax")
        if self.temperature <= 0.0 or self.top_k < 1:
            raise ConfigError("evidence temperature and top_k must be positive")
        if self.pairwise_query_chunk < 1:
            raise ConfigError("pairwise_query_chunk must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260804
    temperature: float = 0.07
    max_steps: int = 32
    logical_physical_batch: int = 128
    physical_microbatch: int = 16
    captions_per_pair: int = 2
    use_generated_unverified: bool = False
    use_masks: bool = False

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ConfigError("training temperature must be positive")
        if self.max_steps < 1 or self.logical_physical_batch < 1:
            raise ConfigError("training steps and logical batch must be positive")
        if self.physical_microbatch < 1 or self.captions_per_pair < 1:
            raise ConfigError("microbatch and captions_per_pair must be positive")
        if self.use_generated_unverified:
            raise ConfigError("generated_unverified rows are forbidden in primary training")
        if self.use_masks:
            raise ConfigError("masks are forbidden in primary retrieval training")


@dataclass(frozen=True)
class QCPRConfig:
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    text: TextConfig = field(default_factory=TextConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    architecture_id: str = "QCPR_V3_TEMPORAL_RETRIEVAL"
    model_revision: str = "model-v3"
    retrieval_dim: int = 512

    def validate(self) -> None:
        if self.retrieval_dim != 512:
            raise ConfigError("retrieval_dim is fixed to 512")
        self.temporal.validate()
        self.text.validate()
        self.evidence.validate()
        self.training.validate()


def validate_config(config: QCPRConfig | Mapping[str, Any]) -> QCPRConfig:
    """Validate a typed config or a strict nested mapping."""
    if isinstance(config, QCPRConfig):
        config.validate()
        return config
    if not isinstance(config, Mapping):
        raise ConfigError("config must be QCPRConfig or a mapping")
    allowed = {"temporal", "text", "evidence", "training", "architecture_id", "model_revision", "retrieval_dim"}
    unknown = set(config) - allowed
    if unknown:
        raise ConfigError(f"unknown top-level config fields: {sorted(unknown)}")
    result_kwargs: dict[str, Any] = {}
    for name, cls in (
        ("temporal", TemporalConfig),
        ("text", TextConfig),
        ("evidence", EvidenceConfig),
        ("training", TrainingConfig),
    ):
        value = config.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"{name} must be a mapping")
        valid = {item.name for item in cls.__dataclass_fields__.values()}
        extra = set(value) - valid
        if extra:
            raise ConfigError(f"unknown {name} fields: {sorted(extra)}")
        result_kwargs[name] = cls(**value)
    for name in ("architecture_id", "model_revision", "retrieval_dim"):
        if name in config:
            result_kwargs[name] = config[name]
    result = QCPRConfig(**result_kwargs)
    result.validate()
    return result


def config_dict(config: QCPRConfig) -> dict[str, Any]:
    validate_config(config)
    return asdict(config)
