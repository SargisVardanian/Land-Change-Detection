"""Strict structural and mask-free validation for Dataset-v2 artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..identities.hashing import sha256_file
from .schemas import ITEM_TYPES, QUERY_SCOPES, SPLITS, VERIFICATION_STATES, PhysicalItem, QueryRecord

FORBIDDEN_KEY_FRAGMENTS = (
    "mask_path",
    "mask_paths",
    "segmentation_path",
    "segmentation_paths",
    "official_label",
    "dense_label",
    "dense_path",
    "label_path",
    "label_paths",
    "object_count",
    "mask_derived",
    "derived_location",
)
TRAINING_VERIFICATION = {"human", "human_rewritten", "human_adjudicated"}


class ValidationError(ValueError):
    """Raised when a record violates a release contract."""


def _key_is_forbidden(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS)


def forbidden_key_hits(value: Any, path: str = "$", *, _hits: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    hits = _hits if _hits is not None else []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _key_is_forbidden(key_text):
                hits.append({"path": child_path, "key": key_text})
            forbidden_key_hits(child, child_path, _hits=hits)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden_key_hits(child, f"{path}[{index}]", _hits=hits)
    return hits


def assert_mask_free(value: Any, *, context: str = "manifest") -> None:
    hits = forbidden_key_hits(value)
    if hits:
        raise ValidationError(f"{context} contains forbidden dense fields: {json.dumps(hits, sort_keys=True)}")


def validate_physical_item(item: PhysicalItem) -> list[str]:
    errors: list[str] = []
    if not item.item_id or not item.source or not item.source_revision:
        errors.append("item_id/source/source_revision must be non-empty")
    if item.item_type not in ITEM_TYPES:
        errors.append(f"unsupported item_type={item.item_type!r}")
    if item.split not in SPLITS:
        errors.append(f"unsupported split={item.split!r}")
    if not item.physical_group_id or not item.scene_id:
        errors.append("physical_group_id/scene_id must be non-empty")
    if item.item_type == "pair" and len(item.frames) != 2:
        errors.append("a physical pair must contain exactly two frames")
    if item.item_type == "sequence" and len(item.frames) < 3:
        errors.append("a physical sequence must contain at least three frames")
    frame_ids = [frame.frame_id for frame in item.frames]
    if len(frame_ids) != len(set(frame_ids)):
        errors.append("frame_id values must be unique within an item")
    for frame in item.frames:
        if not frame.path or not re.fullmatch(r"[0-9a-fA-F]{64}", frame.sha256):
            errors.append(f"invalid frame path/hash for {frame.frame_id!r}")
        if not frame.timestamp:
            errors.append(f"missing timestamp for {frame.frame_id!r}")
    timestamps = [frame.timestamp for frame in item.frames]
    if len(timestamps) != len(set(timestamps)):
        errors.append("frame timestamps must be unique within an item")
    return errors


def _valid_temporal_extent(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    start = value.get("start") or value.get("start_timestamp") or value.get("from")
    end = value.get("end") or value.get("end_timestamp") or value.get("to")
    return bool(str(start or "").strip()) and bool(str(end or "").strip())


def _valid_frame_range(value: Any) -> bool:
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


def validate_query_record(
    query: QueryRecord,
    item_ids: set[str] | None = None,
    *,
    items: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not query.query_id or not query.text.strip():
        errors.append("query_id/text must be non-empty")
    if query.query_scope not in QUERY_SCOPES:
        errors.append(f"unsupported query_scope={query.query_scope!r}")
    if query.verification not in VERIFICATION_STATES:
        errors.append(f"unsupported verification={query.verification!r}")
    if query.split not in SPLITS:
        errors.append(f"unsupported split={query.split!r}")
    if not query.positive_item_ids or query.source_item_id not in query.positive_item_ids:
        errors.append("source_item_id must be one of positive_item_ids")
    if len(query.positive_item_ids) != len(set(query.positive_item_ids)):
        errors.append("positive_item_ids must be unique")
    positive_ids = set(query.positive_item_ids)
    if set(query.graded_relevance) != positive_ids:
        errors.append("graded_relevance keys must exactly match positive_item_ids")
    for item_id, grade in query.graded_relevance.items():
        if grade not in {1, 2, 3}:
            errors.append(f"graded relevance for {item_id!r} must be an integer from 1 to 3")
    if query.temporal_direction not in {"forward", "reverse", "none"}:
        errors.append(f"unsupported temporal_direction={query.temporal_direction!r}")
    if query.training_enabled and query.verification not in TRAINING_VERIFICATION:
        errors.append("unverified/generated/derived text cannot be training-enabled")
    if query.query_scope == "semantic" and len(positive_ids) < 2:
        errors.append("semantic queries must have at least two positive items")
    if query.query_scope == "direction" and query.temporal_direction not in {"forward", "reverse"}:
        errors.append("direction queries require temporal_direction=forward or reverse")
    if query.query_scope == "localized" and not isinstance(query.localized_relation, Mapping):
        errors.append("localized queries require localized_relation metadata")
    if item_ids is not None:
        missing = positive_ids - item_ids
        if missing:
            errors.append(f"positive item IDs are missing from physical registry: {sorted(missing)[:3]}")
    if items is not None:
        source_item = items.get(query.source_item_id)
        if source_item is None:
            errors.append(f"source item ID is missing from physical registry: {query.source_item_id}")
        positive_items = [items[item_id] for item_id in query.positive_item_ids if item_id in items]
        if positive_items:
            splits = {str(item.get("split")) for item in positive_items}
            if query.split not in splits:
                errors.append("query split must match its positive physical items")
            if len(splits) > 1:
                errors.append("positive physical items must share one split")
            if query.training_enabled and not all(bool(item.get("training_enabled")) for item in positive_items):
                errors.append("training-enabled queries require training-enabled positive physical items")
            if query.query_scope == "long_series":
                if source_item is None or str(source_item.get("item_type")) != "sequence":
                    errors.append("long_series queries require a sequence physical item")
                elif len(source_item.get("frames") or []) < 3:
                    errors.append("long_series queries require at least three frames")
                provenance = query.provenance
                if not _valid_temporal_extent(provenance.get("query_temporal_extent")):
                    errors.append("long_series queries require query_temporal_extent start/end")
                if not _valid_frame_range(provenance.get("relevant_frame_range")):
                    errors.append("long_series queries require relevant_frame_range")
    return errors


def validate_rows(records: Iterable[Mapping[str, Any]], *, kind: str) -> list[str]:
    errors: list[str] = []
    for index, raw in enumerate(records, start=1):
        try:
            if kind == "physical":
                item = PhysicalItem.from_dict(raw)
                current = validate_physical_item(item)
            elif kind == "query":
                current = validate_query_record(QueryRecord.from_dict(raw))
            else:
                raise ValueError(f"unknown validation kind={kind}")
        except (KeyError, TypeError, ValueError) as exc:
            current = [f"parse error: {exc}"]
        errors.extend(f"row {index}: {message}" for message in current)
    return errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: JSONL rows must be objects")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)
