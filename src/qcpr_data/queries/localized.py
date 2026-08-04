"""Localized evaluation queries; dense evidence is emitted only as sidecars."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import DenseEvaluationSidecar, QueryRecord


def _relation(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    regions = {
        "top": "upper",
        "bottom": "lower",
        "left": "left",
        "right": "right",
        "center": "center",
        "north": "north",
        "south": "south",
    }
    found = [value for token, value in regions.items() if token in lowered]
    return {"regions": sorted(set(found)), "source": "source_query_text", "evaluation_only": True}


def build_localized_eval_queries(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    dense_by_caption: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    for row in rows:
        source_key = str(row.get("canonical_pair_id") or "")
        item = items.get(source_key)
        if item is None:
            continue
        caption_id = str(row.get("caption_id") or row.get("query_id") or "")
        text = str(row.get("caption") or row.get("text") or "").strip()
        if not text:
            continue
        query = QueryRecord(
            query_id=str(row.get("query_id") or f"{source_key}:localized:{caption_id}"),
            text=text,
            query_scope="localized",
            source_item_id=str(item["item_id"]),
            positive_item_ids=(str(item["item_id"]),),
            graded_relevance={str(item["item_id"]): 3},
            temporal_direction="none",
            localized_relation=_relation(text),
            verification="derived_eval",
            training_enabled=False,
            split=str(item["split"]),
            provenance={"source_caption_id": caption_id, "source_dataset": row.get("dataset_name")},
        )
        queries.append(query.to_dict())
        dense = dense_by_caption.get(caption_id)
        if dense:
            sidecars.append(
                DenseEvaluationSidecar(
                    item_id=str(item["item_id"]),
                    mask_paths=(str(dense["mask_path"]),) if dense.get("mask_path") else (),
                    label_paths=(),
                    derived_attributes={"query_id": query.query_id, "caption_id": caption_id},
                    evaluation_only=True,
                ).to_dict()
            )
    return sorted(queries, key=lambda row: row["query_id"]), sorted(sidecars, key=lambda row: row["item_id"])
