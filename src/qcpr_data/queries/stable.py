"""Stable-scene queries with generic no-change templates excluded."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord

GENERIC_STABLE = {
    "there is no difference",
    "almost nothing has changed",
    "no change has occurred",
    "the two scenes seem identical",
    "the scene is the same as before",
}


def build_stable_queries(
    captions: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for caption in captions:
        if str(caption.get("query_scope") or "") != "generic_no_change":
            continue
        text = " ".join(str(caption.get("text") or "").casefold().split()).strip(" .")
        if text in GENERIC_STABLE:
            continue
        source_key = str(caption.get("canonical_pair_id") or "")
        item = items.get(source_key)
        if item is None:
            continue
        output.append(
            QueryRecord(
                query_id=f"{source_key}:stable:{caption.get('caption_id')}",
                text=str(caption.get("text") or "").strip(),
                query_scope="stable",
                source_item_id=str(item["item_id"]),
                positive_item_ids=(str(item["item_id"]),),
                graded_relevance={str(item["item_id"]): 3},
                temporal_direction="none",
                localized_relation=None,
                verification=str(caption.get("verification_status") or "generated_unverified"),
                training_enabled=False,
                split=str(item["split"]),
                provenance={"generic_no_change_excluded": False},
            ).to_dict()
        )
    return sorted(output, key=lambda row: row["query_id"])
