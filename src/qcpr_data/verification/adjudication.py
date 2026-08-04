"""Adjudication completeness checks."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def validate_adjudication(rows: Iterable[Mapping[str, Any]], expected_ids: set[str]) -> dict[str, Any]:
    decisions = list(rows)
    seen = {str(row.get("audit_row_id")) for row in decisions}
    return {
        "row_count": len(decisions),
        "covers_expected_rows": seen == expected_ids,
        "reconstructed_caption_count": sum(bool(row.get("reconstructed_factual_caption")) for row in decisions),
        "adjudicator_timestamps": all(row.get("adjudicated_at") for row in decisions),
    }
