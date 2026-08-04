"""Build exact-discriminative queries from verified source captions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord
from ..contracts.validation import TRAINING_VERIFICATION
from .direction import infer_direction


def build_exact_queries(
    captions: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    *,
    include_training: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for caption in captions:
        scope = str(caption.get("query_scope") or "")
        if scope != "exact_pair":
            continue
        source_key = str(caption.get("canonical_pair_id") or caption.get("source_item_id") or "")
        item = items.get(source_key)
        if item is None:
            continue
        verification = str(caption.get("verification_status") or "generated_unverified")
        training_enabled = bool(include_training and verification in TRAINING_VERIFICATION and item.get("training_enabled"))
        query = QueryRecord(
            query_id=f"{source_key}:exact:{caption.get('caption_id')}",
            text=str(caption.get("text") or "").strip(),
            query_scope="exact",
            source_item_id=str(item["item_id"]),
            positive_item_ids=(str(item["item_id"]),),
            graded_relevance={str(item["item_id"]): 3},
            temporal_direction=infer_direction(str(caption.get("text") or "")),
            localized_relation=None,
            verification=verification,
            training_enabled=training_enabled,
            split=str(item["split"]),
            provenance={
                "source_caption_id": str(caption.get("caption_id")),
                "source_dataset": str(caption.get("dataset_name") or ""),
                "source_query_scope": scope,
            },
        )
        output.append(query.to_dict())
    return sorted(output, key=lambda row: row["query_id"])
