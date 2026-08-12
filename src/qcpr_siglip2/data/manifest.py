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


def _physical_registry_for_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load the release physical-item registry next to an expanded manifest.

    r19g keeps query records and physical frame records in separate registries.
    This join is deliberately local to the release root and never opens dense
    labels or evaluation sidecars.
    """

    registry_path = path.parent / "registries" / "physical_items.jsonl"
    if not registry_path.is_file():
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(registry_path):
        item_id = item.get("item_id") or item.get("physical_item_id")
        if item_id:
            registry[str(item_id)] = item
    return registry


def _normalize_exact_row(
    row: dict[str, Any], physical_registry: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Normalize legacy and r19g query records to the model batch contract."""

    normalized = dict(row)
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    caption_provenance = row.get("caption_provenance")
    if not isinstance(caption_provenance, dict):
        caption_provenance = {}
    pair_ids = row.get("positive_item_ids")
    pair_id = row.get("canonical_pair_id")
    if not pair_id and isinstance(pair_ids, list) and len(pair_ids) == 1:
        pair_id = pair_ids[0]
    if not pair_id:
        pair_id = row.get("source_item_id") or row.get("source_pair_id")
    if pair_id:
        normalized["canonical_pair_id"] = str(pair_id)
    if "positive_pair_ids" not in normalized and isinstance(pair_ids, list):
        normalized["positive_pair_ids"] = [str(value) for value in pair_ids]
    if "ignored_pair_ids" not in normalized:
        ignored_ids = row.get("ignored_item_ids")
        if isinstance(ignored_ids, list):
            normalized["ignored_pair_ids"] = [str(value) for value in ignored_ids]
        else:
            normalized["ignored_pair_ids"] = []
    normalized.setdefault("caption_id", row.get("query_id") or row.get("canonical_query_id"))
    normalized.setdefault("caption", row.get("text"))
    normalized.setdefault(
        "dataset_name",
        provenance.get("source_dataset") or caption_provenance.get("source_dataset"),
    )
    if normalized.get("query_scope") == "exact":
        normalized["query_scope"] = "exact_pair"
    item = physical_registry.get(str(normalized.get("canonical_pair_id")))
    if item is not None:
        frames = item.get("frames")
        if isinstance(frames, list) and len(frames) >= 2:
            paths = [
                frame.get("native_path") or frame.get("path")
                for frame in frames
                if isinstance(frame, dict)
            ]
            if len(paths) == len(frames) and all(isinstance(value, str) for value in paths):
                normalized["frames"] = paths
                normalized.setdefault("t1_path", paths[0])
                normalized.setdefault("t2_path", paths[1])
    return normalized


def load_exact_core_rows(
    path: str | Path, *, split: str | None = None
) -> list[dict[str, Any]]:
    """Load only human/verified exact rows from LEVIR-MCI and SECOND-CC."""

    manifest_path = Path(path)
    rows = read_jsonl(manifest_path)
    physical_registry = _physical_registry_for_manifest(manifest_path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        normalized = _normalize_exact_row(row, physical_registry)
        _validate_exact_row(normalized)
        selected.append(normalized)
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
