from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any


class RetrievalMode(StrEnum):
    STATIC_REGION = "static_region"
    PAIR_ANALOG = "pair_analog"
    TEXT_BITEMPORAL = "text_bitemporal"
    NOVELTY_SINGLE_IMAGE = "novelty_single_image"
    TRANSITION_CONDITIONED = "transition_conditioned"
    TRAJECTORY = "trajectory"


def _normalize(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


@dataclass(frozen=True)
class RetrievalQuery:
    mode: RetrievalMode
    top_k: int = 5
    text: str | None = None
    item_id: str | None = None
    image_path: str | None = None
    before_image_path: str | None = None
    after_image_path: str | None = None
    region_hint: dict[str, Any] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _normalize(self)


@dataclass(frozen=True)
class RetrievalItem:
    item_id: str
    score: float
    mode: RetrievalMode
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    thumbnail_path: str | None = None
    before_image_path: str | None = None
    after_image_path: str | None = None
    region: dict[str, Any] | None = None
    transition_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _normalize(self)


@dataclass(frozen=True)
class RetrievalResult:
    query: RetrievalQuery
    items: list[RetrievalItem]
    backend_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _normalize(self)


@dataclass(frozen=True)
class RetrievalArtifact:
    mode: RetrievalMode
    result: RetrievalResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> list[RetrievalItem]:
        return self.result.items

    def to_dict(self) -> dict[str, Any]:
        return _normalize(self)
