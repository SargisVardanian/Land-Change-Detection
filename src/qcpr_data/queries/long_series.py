"""Long-series candidate query conversion."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord


def build_long_series_queries(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        sequence_key = str(row.get("sequence_id") or "")
        item = items.get(sequence_key)
        if item is None:
            continue
        text = str(row.get("text") or row.get("output_description") or "").strip()
        if not text:
            continue
        extent = row.get("query_temporal_extent") if isinstance(row.get("query_temporal_extent"), Mapping) else None
        output.append(
            QueryRecord(
                query_id=str(row.get("query_id") or f"{sequence_key}:long_series"),
                text=text,
                query_scope="long_series",
                source_item_id=str(item["item_id"]),
                positive_item_ids=(str(item["item_id"]),),
                graded_relevance={str(item["item_id"]): 3},
                temporal_direction="forward",
                localized_relation=None,
                verification=str(row.get("verification_status") or "generated_unverified"),
                training_enabled=False,
                split=str(item["split"]),
                provenance={
                    "relevant_frame_range": row.get("relevant_frame_range"),
                    "query_temporal_extent": dict(extent) if extent else None,
                    "source_text_provenance": row.get("text_provenance"),
                },
            ).to_dict()
        )
    return sorted(output, key=lambda row: row["query_id"])
