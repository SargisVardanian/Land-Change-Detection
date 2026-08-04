"""Semantic multi-positive query conversion with event IDs kept provenance-only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord


def _grade(value: Any, default: int) -> int:
    """Return a bounded graded-relevance value for a semantic positive."""

    try:
        grade = int(value)
    except (TypeError, ValueError):
        grade = default
    return max(1, min(3, grade))


def _row_grade(row: Mapping[str, Any], item_id: str, *, self_item: bool) -> int:
    explicit = row.get("graded_relevance")
    if isinstance(explicit, Mapping) and item_id in explicit:
        return _grade(explicit[item_id], 3 if self_item else 2)
    if self_item:
        return _grade(row.get("self_relevance_grade"), 3)
    return _grade(row.get("positive_relevance_grade"), 2)


def build_semantic_eval_queries(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one query per source text with all same-group items as positives.

    Semantic groups are keyed by ``(semantic_group_id, split)`` so an external
    grouping error cannot pull positives across release partitions.  Singleton
    groups are intentionally omitted: they are exact-pair supervision, not the
    required semantic multi-positive view.
    """

    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"items": {}, "rows": []}
    )
    for row_index, row in enumerate(rows):
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
        item_id = str(item["item_id"])
        split = str(item.get("split") or "")
        group = groups[(group_id, split)]
        existing = group["items"].get(item_id)
        if existing is None:
            group["items"][item_id] = {
                "grade": _row_grade(row, item_id, self_item=True),
                "row_index": row_index,
            }
        else:
            existing["grade"] = max(existing["grade"], _row_grade(row, item_id, self_item=True))
        group["rows"].append(
            {
                "row": row,
                "item_id": item_id,
                "source_key": source_key,
                "text": text,
                "row_index": row_index,
            }
        )

    output: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for (group_id, split), group in sorted(groups.items()):
        positive_ids = sorted(group["items"])
        if len(positive_ids) < 2:
            continue
        for entry in group["rows"]:
            row = entry["row"]
            item_id = entry["item_id"]
            grades = {
                positive_id: _row_grade(row, positive_id, self_item=positive_id == item_id)
                for positive_id in positive_ids
            }
            query_id = str(row.get("query_id") or f"{entry['source_key']}:semantic:{group_id}")
            if query_id in seen_query_ids:
                query_id = f"{query_id}:row{entry['row_index']}"
            seen_query_ids.add(query_id)
            query = QueryRecord(
                query_id=query_id,
                text=entry["text"],
                query_scope="semantic",
                source_item_id=item_id,
                positive_item_ids=tuple(positive_ids),
                graded_relevance=grades,
                temporal_direction=str(row.get("temporal_direction") or "none"),
                localized_relation=None,
                verification="derived_eval",
                training_enabled=False,
                split=split,
                provenance={
                    "semantic_group_id": group_id,
                    "positive_item_count": len(positive_ids),
                    "source_verification_status": str(row.get("verification_status") or ""),
                    "event_ids_are_provenance_only": True,
                },
            )
            output.append(query.to_dict())
    return sorted(output, key=lambda row: row["query_id"])
