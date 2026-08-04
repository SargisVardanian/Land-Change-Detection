"""Claim-level review helpers."""

from __future__ import annotations

from typing import Any, Mapping


def claim_agreement(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 1.0
    return sum(left.get(key) == right.get(key) for key in keys) / len(keys)
