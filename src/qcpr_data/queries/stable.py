"""Stable-scene query builders.

Generic no-change captions are diagnostic only.  A stable-scene query must
carry independently supported anchors from T1 and T2; the builder will never
turn a generic sentence into a stable query.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.schemas import QueryRecord

GENERIC_STABLE = {
    "there is no difference",
    "there are no differences",
    "almost nothing has changed",
    "no change has occurred",
    "no change is occurred",
    "the two images are the same",
    "the two scenes seem identical",
    "the scene is the same as before",
    "no visible differences exist",
    "there is no change",
}


def _normalise(text: Any) -> str:
    return " ".join(str(text or "").casefold().split()).strip(" .")


def _verification(value: Any) -> str:
    raw = str(value or "")
    return raw if raw in {"human", "human_rewritten", "human_adjudicated", "generated_verified", "generated_unverified", "rule_based_unverified", "derived_eval", "rejected"} else "rule_based_unverified"


def build_stable_queries(
    captions: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build only unique, anchor-backed stable queries.

    Candidate rows may use ``common_atomic_anchors`` (probe output) or
    ``stable_anchors`` (review output).  A row with multiple positives is a
    semantic candidate and is intentionally left to the semantic builder.
    """

    output: list[dict[str, Any]] = []
    for caption in captions:
        anchors = caption.get("common_atomic_anchors") or caption.get("stable_anchors") or []
        if not isinstance(anchors, (list, tuple)) or len(set(map(str, anchors))) < 2:
            continue
        text = str(caption.get("query_text") or caption.get("text") or "").strip()
        if not text or _normalise(text) in GENERIC_STABLE:
            continue
        source_key = str(caption.get("canonical_pair_id") or caption.get("source_item_id") or "")
        item = items.get(source_key)
        if item is None:
            continue
        positive_ids = [str(value) for value in caption.get("stable_positive_item_ids", [item["item_id"]])]
        if len(positive_ids) != 1:
            continue
        identifiability = float(caption.get("identifiability_score") or 0.0)
        if identifiability < 0.5:
            continue
        item_id = str(item["item_id"])
        output.append(
            QueryRecord(
                query_id=str(caption.get("query_id") or caption.get("candidate_id") or f"{source_key}:stable"),
                text=text,
                query_scope="stable",
                source_item_id=item_id,
                positive_item_ids=(item_id,),
                graded_relevance={item_id: 3},
                temporal_direction="none",
                localized_relation=None,
                verification=_verification(caption.get("verification")),
                training_enabled=False,
                split=str(item["split"]),
                provenance={
                    "stable_anchors": sorted(set(map(str, anchors))),
                    "identifiability_score": identifiability,
                    "independent_t1_claims": caption.get("independent_t1_claims", []),
                    "independent_t2_claims": caption.get("independent_t2_claims", []),
                    "masks_used_for_text": False,
                    "human_review_required": True,
                },
            ).to_dict()
        )
    return sorted(output, key=lambda row: row["query_id"])
