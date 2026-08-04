"""Shared source adapter primitives.

Adapters accept already acquired, hashed registries.  They do not decide
whether a source is licensed, split-safe, or training-enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..contracts.schemas import FrameRecord, PhysicalItem


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def _row_name(row: Mapping[str, Any]) -> str:
    return str(row.get("canonical_pair_id") or row.get("pair_id") or row.get("sequence_id") or row.get("item_id"))


def _frame_from_value(item_id: str, index: int, raw: Mapping[str, Any]) -> FrameRecord:
    return FrameRecord(
        frame_id=str(raw.get("frame_id") or f"{item_id}:frame:{index}"),
        path=str(raw.get("path") or raw.get("image_path") or ""),
        sha256=str(raw.get("sha256") or ""),
        timestamp=str(raw.get("timestamp") or raw.get("date") or f"t{index + 1}"),
        sensor=str(raw["sensor"]) if raw.get("sensor") is not None else None,
        gsd=float(raw["gsd"]) if raw.get("gsd") is not None else None,
        width=int(raw["width"]) if raw.get("width") is not None else None,
        height=int(raw["height"]) if raw.get("height") is not None else None,
    )


def normalize_pair_row(
    row: Mapping[str, Any],
    *,
    source: str,
    source_revision: str,
    split: str,
    training_enabled: bool,
    quality_status: str,
    item_id_value: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> PhysicalItem:
    stable_id = item_id_value or _row_name(row)
    item_id = stable_id if stable_id.casefold().startswith(f"{source}:".casefold()) else f"{source}:{source_revision}:{stable_id}"
    raw_frames = row.get("frames")
    if not raw_frames:
        raw_frames = [
            {"path": row.get("t1_path"), "sha256": row.get("t1_sha256"), "timestamp": "t1"},
            {"path": row.get("t2_path"), "sha256": row.get("t2_sha256"), "timestamp": "t2"},
        ]
    frames = tuple(_frame_from_value(item_id, index, frame) for index, frame in enumerate(raw_frames))
    scene_id = str(row.get("scene_id") or row.get("source_scene_group_id") or stable_id)
    return PhysicalItem(
        item_id=item_id,
        item_type="pair",
        source=source,
        source_revision=source_revision,
        physical_group_id=str(row.get("physical_group_id") or scene_id),
        scene_id=scene_id,
        event_id=str(row.get("event_id") or row.get("source_event_id")) if (row.get("event_id") is not None or row.get("source_event_id") is not None) else None,
        frames=frames,
        split=split,
        training_enabled=training_enabled,
        quality_status=quality_status,
        provenance=dict(provenance or {}),
    )


def normalize_sequence_row(
    row: Mapping[str, Any],
    *,
    source: str,
    source_revision: str,
    split: str,
    training_enabled: bool,
    quality_status: str,
    item_id_value: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> PhysicalItem:
    stable_id = item_id_value or _row_name(row)
    item_id = stable_id if stable_id.casefold().startswith(f"{source}:".casefold()) else f"{source}:{source_revision}:{stable_id}"
    frames = tuple(_frame_from_value(item_id, index, frame) for index, frame in enumerate(row.get("frames", [])))
    scene_id = str(row.get("scene_id") or row.get("source_scene_group_id") or stable_id)
    return PhysicalItem(
        item_id=item_id,
        item_type="sequence",
        source=source,
        source_revision=source_revision,
        physical_group_id=str(row.get("physical_group_id") or scene_id),
        scene_id=scene_id,
        event_id=str(row.get("event_id") or row.get("source_event_id")) if (row.get("event_id") is not None or row.get("source_event_id") is not None) else None,
        frames=frames,
        split=split,
        training_enabled=training_enabled,
        quality_status=quality_status,
        provenance=dict(provenance or {}),
    )


@dataclass(frozen=True)
class RegistrySourceAdapter:
    """Thin adapter around a source registry; release policy is injected."""

    source_name: str
    source_revision: str = "registry"

    def rows(self, registry_path: Path) -> Iterator[dict[str, Any]]:
        for row in iter_jsonl(registry_path):
            row_source = str(row.get("source_dataset") or row.get("dataset_name") or "").casefold()
            if row_source == self.source_name.casefold():
                yield row

    def pairs(
        self,
        registry_path: Path,
        *,
        split_resolver: Callable[[Mapping[str, Any]], str],
        training_enabled: bool = False,
        quality_status: str = "PHYSICAL_ONLY",
    ) -> Iterator[PhysicalItem]:
        for row in self.rows(registry_path):
            yield normalize_pair_row(
                row,
                source=self.source_name,
                source_revision=self.source_revision,
                split=split_resolver(row),
                training_enabled=training_enabled,
                quality_status=quality_status,
                item_id_value=str(row.get("canonical_pair_id") or row.get("pair_id")),
                provenance={"adapter": self.source_name, "source_split": row.get("split")},
            )
