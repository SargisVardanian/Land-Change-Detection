"""Promotion gate for source text."""

from __future__ import annotations

from typing import Any, Mapping


def can_promote_text(
    verification: str,
    *,
    reviewer_gate: Mapping[str, Any] | None = None,
    allow_verified_state: bool = True,
) -> bool:
    if not allow_verified_state or verification not in {"human", "human_rewritten", "human_adjudicated"}:
        return False
    if verification == "human_adjudicated":
        return bool(reviewer_gate and reviewer_gate.get("complete") and reviewer_gate.get("distinct_reviewers"))
    return True
