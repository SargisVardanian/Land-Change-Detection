"""Semantic multi-positive query conversion with event IDs kept provenance-only."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord


def build_semantic_eval_queries(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        source_key = str(row.get("canonical_pair_id") or row.get("pair_id") or "")
        item = items.get(source_key)
        if item is None:
            continue
        group_id = str(row.get("semantic_group_id") or "")
        if (
            not group_id
            or group_id.casefold().startswith(("event:", "event_", "event/"))
            or row.get("event_id") is not None
            or row.get("source_event_id") is not None
        ):
            continue
        text = str(row.get("text") or row.get("normalized_text") or "").strip()
        if not text:
            continue
        query = QueryRecord(
            query_id=str(row.get("query_id") or f"{source_key}:semantic"),
            text=text,
            query_scope="semantic",
            source_item_id=str(item["item_id"]),
            positive_item_ids=(str(item["item_id"]),),
            graded_relevance={str(item["item_id"]): int(row.get("self_relevance_grade") or 3)},
            temporal_direction=str(row.get("temporal_direction") or "none"),
            localized_relation=None,
            verification="derived_eval",
            training_enabled=False,
            split=str(item["split"]),
            provenance={
                "semantic_group_id": group_id,
                "source_verification_status": str(row.get("verification_status") or ""),
                "event_ids_are_provenance_only": True,
            },
        )
        output.append(query.to_dict())
    return sorted(output, key=lambda row: row["query_id"])
