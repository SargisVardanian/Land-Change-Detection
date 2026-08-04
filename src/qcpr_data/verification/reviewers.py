"""Validation of independently recorded reviewer exports."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def validate_review_pair(
    reviewer_a: Iterable[Mapping[str, Any]],
    reviewer_b: Iterable[Mapping[str, Any]],
    *,
    reviewer_a_id: str,
    reviewer_b_id: str,
) -> dict[str, Any]:
    left = list(reviewer_a)
    right = list(reviewer_b)
    ids_left = {str(row.get("audit_row_id")) for row in left}
    ids_right = {str(row.get("audit_row_id")) for row in right}
    complete = bool(left and ids_left == ids_right and len(ids_left) == len(left))
    distinct = reviewer_a_id and reviewer_b_id and reviewer_a_id != reviewer_b_id
    timestamps = all(row.get("reviewed_at") for row in left + right)
    return {
        "rows_a": len(left),
        "rows_b": len(right),
        "complete": complete,
        "distinct_reviewers": bool(distinct),
        "timestamps_present": timestamps,
        "independence_attested": all(bool(row.get("independence_attestation")) for row in left + right),
        "decision_counts_a": dict(Counter(str(row.get("decision")) for row in left)),
        "decision_counts_b": dict(Counter(str(row.get("decision")) for row in right)),
    }
