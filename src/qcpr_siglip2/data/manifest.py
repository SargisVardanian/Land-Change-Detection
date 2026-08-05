"""Strict, mask-free manifest helpers for the approved exact retrieval core."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ALLOWED_DATASETS = frozenset({"levir_mci", "second_cc"})
_FORBIDDEN_VERIFICATION = frozenset({"generated_unverified", "derived_eval"})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL and reject malformed or non-object records."""

    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {manifest_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"manifest row at {manifest_path}:{line_number} is not an object"
                )
            rows.append(row)
    return rows


def _validate_exact_row(row: dict[str, Any]) -> None:
    dataset_name = row.get("dataset_name")
    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(f"unexpected source in exact core: {dataset_name!r}")
    required = ("canonical_pair_id", "caption_id", "t1_path", "t2_path", "caption")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"exact row missing required fields: {missing}")
    if bool(row.get("is_generated", False)):
        raise ValueError("generated text is not allowed in the primary exact core")
    if str(row.get("verification", "human")) in _FORBIDDEN_VERIFICATION:
        raise ValueError(
            "unverified/derived text is not allowed in the primary exact core"
        )
    for path_field in ("t1_path", "t2_path"):
        path_value = row[path_field]
        if not isinstance(path_value, str):
            raise TypeError(f"{path_field} must be a string")


def load_exact_core_rows(
    path: str | Path, *, split: str | None = None
) -> list[dict[str, Any]]:
    """Load only human/verified exact rows from LEVIR-MCI and SECOND-CC."""

    rows = read_jsonl(path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        _validate_exact_row(row)
        selected.append(row)
    if not selected:
        raise ValueError(f"empty exact core manifest: {path}")
    return selected


def load_exact_pair_rows(
    path: str | Path, *, split: str | None = None
) -> list[dict[str, Any]]:
    """Load the exact-pair subset, excluding generic no-change diagnostics."""

    selected = [
        row
        for row in load_exact_core_rows(path, split=split)
        if row.get("query_scope") == "exact_pair"
    ]
    if not selected:
        raise ValueError(f"manifest has no exact_pair rows: {path}")
    return selected


def group_rows_by_pair(
    rows: Iterable[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Group rows while preserving first-seen physical-pair order."""

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        pair_id = row.get("canonical_pair_id")
        if not pair_id:
            raise ValueError("row is missing canonical_pair_id")
        groups.setdefault(str(pair_id), []).append(row)
    return groups


def paired_caption_rows(
    rows: Iterable[dict[str, Any]], *, captions_per_pair: int = 2
) -> list[dict[str, Any]]:
    """Select a fixed number of captions per pair without changing pair weights."""

    if captions_per_pair <= 0:
        raise ValueError("captions_per_pair must be positive")
    selected: list[dict[str, Any]] = []
    for pair_id, group in group_rows_by_pair(rows).items():
        if len(group) < captions_per_pair:
            raise ValueError(f"pair {pair_id} has only {len(group)} captions")
        selected.extend(group[:captions_per_pair])
    return selected


def ordered_id_sha256(rows: Iterable[dict[str, Any]], field: str) -> str:
    """Hash an ordered ID sequence with an unambiguous line delimiter."""

    values = [str(row[field]) for row in rows]
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()
