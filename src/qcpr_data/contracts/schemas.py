"""Canonical, JSON-compatible Dataset-v2 records.

The records deliberately contain no dataset policy.  Split assignment,
training promotion, and source eligibility are applied by release builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

SPLITS = ("train", "development", "test")
ITEM_TYPES = ("pair", "sequence")
QUERY_SCOPES = ("exact", "semantic", "localized", "direction", "stable", "long_series")
VERIFICATION_STATES = (
    "human",
    "human_rewritten",
    "human_adjudicated",
    "generated_verified",
    "generated_unverified",
    "rule_based_unverified",
    "derived_eval",
    "rejected",
)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    path: str
    sha256: str
    timestamp: str
    sensor: str | None = None
    gsd: float | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameRecord":
        return cls(
            frame_id=str(value["frame_id"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            timestamp=str(value["timestamp"]),
            sensor=_optional_string(value.get("sensor")),
            gsd=float(value["gsd"]) if value.get("gsd") is not None else None,
            width=int(value["width"]) if value.get("width") is not None else None,
            height=int(value["height"]) if value.get("height") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "path": self.path,
            "sha256": self.sha256,
            "timestamp": self.timestamp,
            "sensor": self.sensor,
            "gsd": self.gsd,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PhysicalItem:
    item_id: str
    item_type: str
    source: str
    source_revision: str
    physical_group_id: str
    scene_id: str
    event_id: str | None
    frames: tuple[FrameRecord, ...]
    split: str
    training_enabled: bool
    quality_status: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalItem":
        return cls(
            item_id=str(value["item_id"]),
            item_type=str(value["item_type"]),
            source=str(value["source"]),
            source_revision=str(value["source_revision"]),
            physical_group_id=str(value["physical_group_id"]),
            scene_id=str(value["scene_id"]),
            event_id=_optional_string(value.get("event_id")),
            frames=tuple(FrameRecord.from_dict(frame) for frame in value["frames"]),
            split=str(value["split"]),
            training_enabled=bool(value["training_enabled"]),
            quality_status=str(value["quality_status"]),
            provenance=dict(value.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "source": self.source,
            "source_revision": self.source_revision,
            "physical_group_id": self.physical_group_id,
            "scene_id": self.scene_id,
            "event_id": self.event_id,
            "frames": [frame.to_dict() for frame in self.frames],
            "split": self.split,
            "training_enabled": self.training_enabled,
            "quality_status": self.quality_status,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str
    query_scope: str
    source_item_id: str
    positive_item_ids: tuple[str, ...]
    graded_relevance: Mapping[str, int]
    temporal_direction: str
    localized_relation: Mapping[str, Any] | None
    verification: str
    training_enabled: bool
    split: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryRecord":
        return cls(
            query_id=str(value["query_id"]),
            text=str(value["text"]),
            query_scope=str(value["query_scope"]),
            source_item_id=str(value["source_item_id"]),
            positive_item_ids=tuple(str(item) for item in value["positive_item_ids"]),
            graded_relevance={str(k): int(v) for k, v in dict(value.get("graded_relevance") or {}).items()},
            temporal_direction=str(value.get("temporal_direction") or "none"),
            localized_relation=dict(value["localized_relation"]) if value.get("localized_relation") else None,
            verification=str(value["verification"]),
            training_enabled=bool(value["training_enabled"]),
            split=str(value["split"]),
            provenance=dict(value.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "query_scope": self.query_scope,
            "source_item_id": self.source_item_id,
            "positive_item_ids": list(self.positive_item_ids),
            "graded_relevance": dict(self.graded_relevance),
            "temporal_direction": self.temporal_direction,
            "localized_relation": dict(self.localized_relation) if self.localized_relation else None,
            "verification": self.verification,
            "training_enabled": self.training_enabled,
            "split": self.split,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class DenseEvaluationSidecar:
    item_id: str
    mask_paths: tuple[str, ...] = ()
    label_paths: tuple[str, ...] = ()
    derived_attributes: Mapping[str, Any] = field(default_factory=dict)
    evaluation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "mask_paths": list(self.mask_paths),
            "label_paths": list(self.label_paths),
            "derived_attributes": dict(self.derived_attributes),
            "evaluation_only": self.evaluation_only,
        }
