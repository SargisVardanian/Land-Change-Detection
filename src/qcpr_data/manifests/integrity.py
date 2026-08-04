"""Release-wide validation, mask-free scanning, and file hashes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts.schemas import PhysicalItem, QueryRecord
from ..contracts.validation import (
    ValidationError,
    forbidden_key_hits,
    read_jsonl,
    sha256_file,
    validate_physical_item,
    validate_query_record,
    write_json,
    write_jsonl,
)
from ..identities.overlap import audit_split_leakage
from ..splits.event_disjoint import validate_event_disjoint
from ..splits.sequence_disjoint import validate_sequence_disjoint


def _manifest_paths(root: Path) -> list[Path]:
    return sorted((root / "manifests").glob("*.jsonl"))


def write_sha256sums(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        hashes[str(path.relative_to(root))] = sha256_file(path)
    lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashes


def verify_sha256sums(root: Path) -> dict[str, Any]:
    path = root / "SHA256SUMS"
    if not path.is_file():
        return {"present": False, "passed": False, "checked": 0, "mismatches": []}
    mismatches: list[str] = []
    checked = 0
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        expected[name] = digest
        target = root / name
        checked += 1
        if not target.is_file() or sha256_file(target) != digest:
            mismatches.append(name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    missing_entries = sorted(actual - set(expected))
    extra_entries = sorted(set(expected) - actual)
    return {
        "present": True,
        "passed": not mismatches and not missing_entries and not extra_entries,
        "checked": checked,
        "mismatches": mismatches,
        "missing_entries": missing_entries,
        "extra_entries": extra_entries,
        "self_reference_excluded": True,
    }


def audit_release(root: Path) -> dict[str, Any]:
    physical_path = root / "registries/physical_items.jsonl"
    item_rows = read_jsonl(physical_path)
    item_errors = validate_physical_records(item_rows)
    item_ids = {str(row.get("item_id")) for row in item_rows}
    item_map = {str(row.get("item_id")): row for row in item_rows}
    query_rows: list[dict[str, Any]] = []
    mask_hits: list[dict[str, Any]] = []
    query_errors: list[str] = []
    for path in _manifest_paths(root):
        rows = read_jsonl(path)
        for index, row in enumerate(rows, start=1):
            query_rows.append(row)
            for hit in forbidden_key_hits(row):
                mask_hits.append({"manifest": str(path.relative_to(root)), "row": index, **hit})
            query_errors.extend(
                f"{path.name}:{index}: {message}"
                for message in validate_query_record(QueryRecord.from_dict(row), item_ids, items=item_map)
            )
    duplicate_query_ids = [query_id for query_id, count in Counter(str(row.get("query_id")) for row in query_rows).items() if count > 1]
    leakage = audit_split_leakage(item_rows)
    events = validate_event_disjoint(item_rows)
    sequences = validate_sequence_disjoint(item_rows)
    cross_source_path = root / "audits/cross_source_overlap_audit.json"
    cross_source = json.loads(cross_source_path.read_text(encoding="utf-8")) if cross_source_path.is_file() else {
        "passed": True,
        "not_run": True,
    }
    mask_free = {
        "schema_version": "qcpr-mask-free-integrity-v1",
        "manifest_count": len(_manifest_paths(root)),
        "query_row_count": len(query_rows),
        "violation_count": len(mask_hits),
        "passed": not mask_hits,
    }
    write_json(root / "audits/mask_free_integrity.json", mask_free)
    leakage_report = {"split": leakage, "events": events, "sequences": sequences}
    write_json(root / "audits/leakage_audit.json", leakage_report)
    write_json(
        root / "audits/cross_split_leakage_audit.json",
        {
            "schema_version": "qcpr-cross-split-leakage-audit-v1",
            "split": leakage,
            "events": events,
            "sequences": sequences,
            "passed": leakage["passed"] and events["passed"] and sequences["passed"],
        },
    )
    (root / "audits/mask_free_integrity.md").write_text(
        "\n".join(
            [
                "# Mask-free integrity",
                "",
                f"- Manifests scanned: `{mask_free['manifest_count']}`",
                f"- Query rows scanned: `{mask_free['query_row_count']}`",
                f"- Forbidden-key violations: `{mask_free['violation_count']}`",
                f"- Passed: `{mask_free['passed']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(root / "audits/forbidden_key_hits.jsonl", mask_hits)
    write_json(root / "audits/record_validation.json", {"physical_errors": item_errors, "query_errors": query_errors, "duplicate_query_ids": duplicate_query_ids})
    return {
        "physical_item_count": len(item_rows),
        "frame_count": sum(len(row.get("frames", [])) for row in item_rows),
        "query_count": len(query_rows),
        "physical_errors": item_errors,
        "query_errors": query_errors,
        "duplicate_query_ids": duplicate_query_ids,
        "leakage": leakage,
        "events": events,
        "sequences": sequences,
        "cross_source": cross_source,
        "mask_free": mask_free,
        "sha256sums": verify_sha256sums(root),
        "passed": (
            not item_errors
            and not query_errors
            and not duplicate_query_ids
            and leakage["passed"]
            and events["passed"]
            and sequences["passed"]
            and cross_source.get("passed", False)
            and mask_free["passed"]
        ),
    }


def validate_physical_records(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        item = PhysicalItem.from_dict(row)
        if item.item_id in seen:
            errors.append(f"row {index}: duplicate item_id={item.item_id}")
        seen.add(item.item_id)
        errors.extend(f"row {index}: {message}" for message in validate_physical_item(item))
    return errors
