"""Configurable bounded mixture sampling for Dataset-v2 training manifests."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class MixturePolicy:
    human_weight: float = 1.0
    verified_generated_weight: float = 0.5
    unverified_generated_weight: float = 0.3
    generic_no_change_weight: float = 0.1
    synthetic_max_fraction: float = 0.35

def sample_weight(row: dict, policy: MixturePolicy = MixturePolicy()) -> float:
    if row.get("query_scope") == "generic_no_change": return policy.generic_no_change_weight
    if not row.get("is_generated", False): return policy.human_weight
    if row.get("verification_status") == "verified": return policy.verified_generated_weight
    return policy.unverified_generated_weight

def validate_batch(rows: Iterable[dict], policy: MixturePolicy = MixturePolicy()) -> None:
    rows=list(rows)
    if not rows: raise ValueError("empty logical batch")
    pair_ids=[str(row.get("canonical_pair_id")) for row in rows]
    if len(pair_ids) != len(set(pair_ids)): raise ValueError("physical pair may occur once per logical batch")
    synthetic=sum(bool(row.get("is_synthetic",False)) for row in rows)/len(rows)
    if synthetic > policy.synthetic_max_fraction: raise ValueError("synthetic fraction exceeds configured bound")
    if any(not sample_weight(row,policy)>0 for row in rows): raise ValueError("nonpositive configured sampling weight")
