"""Long-series query conversion with explicit temporal annotation gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable

from ..contracts.schemas import QueryRecord


def _has_temporal_extent(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    start = value.get("start") or value.get("start_timestamp") or value.get("from")
    end = value.get("end") or value.get("end_timestamp") or value.get("to")
    return bool(str(start or "").strip()) and bool(str(end or "").strip())


def _has_frame_range(value: Any) -> bool:
    if isinstance(value, Mapping):
        start = value.get("start")
        if start is None:
            start = value.get("start_frame")
        if start is None:
            start = value.get("first")
        end = value.get("end")
        if end is None:
            end = value.get("end_frame")
        if end is None:
            end = value.get("last")
        return start is not None and end is not None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) >= 2
    return False


def build_long_series_queries(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert only temporally annotated, sequence-backed query candidates.

    Generated descriptions without an independently recorded temporal extent,
    relevant frame range, and explicit direction remain review artifacts and
    must not enter the long-series retrieval manifest.
    """

    output: list[dict[str, Any]] = []
    for row in rows:
        sequence_key = str(row.get("sequence_id") or "")
        item = items.get(sequence_key)
        if item is None or str(item.get("item_type")) != "sequence":
            continue
        frames = item.get("frames") or []
        if len(frames) < 3:
            continue
        text = str(row.get("text") or row.get("output_description") or "").strip()
        if not text:
            continue
        extent = row.get("query_temporal_extent")
        frame_range = row.get("relevant_frame_range")
        if not _has_temporal_extent(extent) or not _has_frame_range(frame_range):
            continue
        direction_value = row.get("temporal_direction")
        if direction_value is None:
            direction_value = row.get("direction")
        direction = str(direction_value or "").strip().casefold()
        if direction not in {"forward", "reverse", "none"}:
            continue
        item_id = str(item["item_id"])
        output.append(
            QueryRecord(
                query_id=str(row.get("query_id") or f"{sequence_key}:long_series"),
                text=text,
                query_scope="long_series",
                source_item_id=item_id,
                positive_item_ids=(item_id,),
                graded_relevance={item_id: 3},
                temporal_direction=direction,
                localized_relation=None,
                verification=str(row.get("verification_status") or "generated_unverified"),
                training_enabled=False,
                split=str(item["split"]),
                provenance={
                    "relevant_frame_range": frame_range,
                    "query_temporal_extent": dict(extent),
                    "change_onset": row.get("change_onset"),
                    "change_duration": row.get("change_duration"),
                    "spatial_evidence_per_time": row.get("spatial_evidence_per_time"),
                    "source_text_provenance": row.get("text_provenance"),
                },
            ).to_dict()
        )
    return sorted(output, key=lambda row: row["query_id"])
