"""Schema-validated model-side data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, cast

import torch
from torch import Tensor

QueryScope = Literal["exact", "semantic", "localized", "direction", "stable", "long_series"]
Verification = Literal["human", "human_rewritten", "generated_verified", "generated_unverified", "derived_eval"]


@dataclass(frozen=True)
class FrameRecord:
    path: str
    sha256: str
    timestamp: str | None = None
    sensor: str | None = None
    gsd: float | None = None


@dataclass(frozen=True)
class RetrievalItem:
    item_id: str
    item_type: Literal["pair", "sequence"]
    frames: tuple[FrameRecord, ...]
    split: Literal["train", "development", "test"]
    source: str
    physical_group_id: str
    event_id: str | None = None
    training_enabled: bool = False


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str
    query_scope: QueryScope
    positive_item_ids: tuple[str, ...]
    graded_relevance: Mapping[str, int]
    verification: Verification
    split: Literal["train", "development", "test"]
    temporal_direction: Literal["forward", "reverse", "none"] = "none"
    localized_relation: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TemporalMetadata:
    timestamps: Tensor
    delta_times: Tensor
    frame_ids: Tensor
    sensor_ids: Tensor | None = None
    gsd: Tensor | None = None
    metadata_missing: Tensor | None = None


@dataclass(frozen=True)
class VisualTokenBatch:
    dense_tokens: Tensor
    coordinates: Tensor
    frame_mask: Tensor
    token_mask: Tensor
    temporal: TemporalMetadata
    sequence_cls: Tensor | None = None
    frame_cls: Tensor | None = None
    change_tokens: Tensor | None = None

    def validate(self) -> None:
        if self.dense_tokens.ndim != 4:
            raise ValueError("dense_tokens must be [B,T,N,D]")
        b, t, n, d = self.dense_tokens.shape
        if self.coordinates.shape != (b, n, 2):
            raise ValueError("coordinates must be [B,N,2]")
        if self.frame_mask.shape != (b, t) or self.frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean [B,T]")
        if self.token_mask.shape != (b, t, n) or self.token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean [B,T,N]")
        if self.temporal.timestamps.shape != (b, t):
            raise ValueError("timestamps must be [B,T]")
        if self.temporal.delta_times.shape != (b, t):
            raise ValueError("delta_times must be [B,T]")
        if self.temporal.frame_ids.shape != (b, t):
            raise ValueError("frame_ids must be [B,T]")
        if self.sequence_cls is not None and self.sequence_cls.shape != (b, d):
            raise ValueError("sequence_cls must be [B,D]")
        if self.frame_cls is not None and self.frame_cls.shape != (b, t, d):
            raise ValueError("frame_cls must be [B,T,D]")
        if self.change_tokens is not None and (
            self.change_tokens.ndim != 3 or self.change_tokens.shape[0] != b or self.change_tokens.shape[2] != d
        ):
            raise ValueError("change_tokens must be [B,K,D]")
        if not self.dense_tokens.is_floating_point():
            raise TypeError("dense_tokens must be floating point")

    @property
    def batch_size(self) -> int:
        return int(self.dense_tokens.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.dense_tokens.shape[-1])

    @property
    def token_count(self) -> int:
        return int(self.dense_tokens.shape[1] * self.dense_tokens.shape[2])


@dataclass(frozen=True)
class QueryTokenBatch:
    text_cls: Tensor
    text_tokens: Tensor
    text_mask: Tensor

    def validate(self) -> None:
        if self.text_cls.ndim != 2 or self.text_tokens.ndim != 3 or self.text_mask.ndim != 2:
            raise ValueError("query tensors must be [B,D], [B,L,D], [B,L]")
        if self.text_cls.shape[0] != self.text_tokens.shape[0] or self.text_mask.shape != self.text_tokens.shape[:2]:
            raise ValueError("query batch dimensions do not agree")
        if self.text_cls.shape[1] != self.text_tokens.shape[2]:
            raise ValueError("text_cls and text_tokens dimensions do not agree")
        if self.text_mask.dtype != torch.bool:
            raise TypeError("text_mask must be boolean")


_FORBIDDEN_TRAINING_TERMS = (
    "mask",
    "dense_label",
    "official_label",
    "semantic_map",
    "binary_change_map",
)


def assert_mask_free_record(record: Mapping[str, Any]) -> None:
    """Reject dense/mask paths recursively in primary training records."""
    def visit(value: Any, key_path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).casefold()
                if any(term in lowered for term in _FORBIDDEN_TRAINING_TERMS):
                    raise ValueError(f"mask/dense field is forbidden: {key_path}.{key}")
                visit(child, f"{key_path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                visit(child, f"{key_path}[{index}]")
        elif isinstance(value, str):
            lowered = value.casefold()
            if any(term in lowered for term in _FORBIDDEN_TRAINING_TERMS) and ("/" in value or "." in value):
                raise ValueError(f"mask/dense path is forbidden: {key_path}")
    visit(record, "$")


def _require(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise ValueError(f"missing required field: {name}")
    return record[name]


def validate_item_record(record: Mapping[str, Any]) -> RetrievalItem:
    frames_raw = _require(record, "frames")
    if not isinstance(frames_raw, Sequence) or len(frames_raw) < 2:
        raise ValueError("item.frames must contain at least two frames")
    frames = tuple(
        FrameRecord(
            path=str(_require(frame, "path")),
            sha256=str(_require(frame, "sha256")),
            timestamp=frame.get("timestamp"),
            sensor=frame.get("sensor"),
            gsd=frame.get("gsd"),
        )
        for frame in frames_raw
    )
    item = RetrievalItem(
        item_id=str(_require(record, "item_id")),
        item_type=cast(Literal["pair", "sequence"], str(_require(record, "item_type"))),
        frames=frames,
        split=cast(Literal["train", "development", "test"], str(_require(record, "split"))),
        source=str(_require(record, "source")),
        physical_group_id=str(_require(record, "physical_group_id")),
        event_id=record.get("event_id"),
        training_enabled=bool(_require(record, "training_enabled")),
    )
    if item.item_type not in {"pair", "sequence"}:
        raise ValueError("item_type must be pair or sequence")
    if item.split not in {"train", "development", "test"}:
        raise ValueError("invalid item split")
    return item


def validate_query_record(record: Mapping[str, Any], *, primary_training: bool = False) -> QueryRecord:
    positives = tuple(str(item_id) for item_id in _require(record, "positive_item_ids"))
    if not positives:
        raise ValueError("positive_item_ids must be non-empty")
    graded = {str(key): int(value) for key, value in dict(_require(record, "graded_relevance")).items()}
    if any(value < 0 or value > 3 for value in graded.values()):
        raise ValueError("graded relevance must be in [0,3]")
    query = QueryRecord(
        query_id=str(_require(record, "query_id")),
        text=str(_require(record, "text")),
        query_scope=cast(QueryScope, str(_require(record, "query_scope"))),
        positive_item_ids=positives,
        graded_relevance=graded,
        verification=cast(Verification, str(_require(record, "verification"))),
        split=cast(Literal["train", "development", "test"], str(_require(record, "split"))),
        temporal_direction=cast(Literal["forward", "reverse", "none"], str(record.get("temporal_direction", "none"))),
        localized_relation=record.get("localized_relation"),
    )
    if query.query_scope not in {"exact", "semantic", "localized", "direction", "stable", "long_series"}:
        raise ValueError("invalid query_scope")
    if query.verification not in {"human", "human_rewritten", "generated_verified", "generated_unverified", "derived_eval"}:
        raise ValueError("invalid verification")
    if query.split not in {"train", "development", "test"}:
        raise ValueError("invalid query split")
    if primary_training:
        assert_mask_free_record(record)
        if query.verification == "generated_unverified":
            raise ValueError("generated_unverified query cannot enter primary training")
    return query
