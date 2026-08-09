#!/usr/bin/env python3
"""Run loader, schema, decode, split-reference, and mask-free checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_contract_package() -> tuple[Any, Any, Any, Any, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from qcpr_data.contracts.schemas import PhysicalItem, QueryRecord
    from qcpr_data.contracts.validation import (
        forbidden_key_hits,
        read_jsonl,
        validate_physical_item,
        validate_query_record,
    )
    from qcpr_data.manifests.integrity import verify_sha256sums

    return PhysicalItem, QueryRecord, forbidden_key_hits, read_jsonl, (validate_physical_item, validate_query_record, verify_sha256sums)


def _decode_sample(items: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment-specific
        return [{"path": "<PIL>", "error": f"{type(exc).__name__}: {exc}"}]
    checked = 0
    for item in items:
        for frame in item.get("frames", []):
            if checked >= limit:
                return failures
            checked += 1
            path = Path(str(frame.get("path") or ""))
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                failures.append({"path": str(path), "error": type(exc).__name__})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-sample", type=int, default=256)
    args = parser.parse_args()
    root = args.release
    output = args.output
    if output is not None:
        try:
            output.resolve().relative_to(root.resolve())
        except ValueError:
            pass
        else:
            parser.error(
                "--output must be outside --release; immutable releases are read-only"
            )
    PhysicalItem, QueryRecord, forbidden_key_hits, read_jsonl, validators = _load_contract_package()
    validate_physical_item, validate_query_record, verify_sha256sums = validators

    physical_rows = read_jsonl(root / "registries/physical_items.jsonl")
    items = {str(row.get("item_id")): row for row in physical_rows}
    physical_errors: list[str] = []
    for index, row in enumerate(physical_rows, start=1):
        try:
            physical_errors.extend(
                f"physical:{index}: {message}"
                for message in validate_physical_item(PhysicalItem.from_dict(row))
            )
        except (KeyError, TypeError, ValueError) as exc:
            physical_errors.append(f"physical:{index}: parse error: {exc}")

    manifests = sorted((root / "manifests").glob("*.jsonl"))
    query_rows: list[dict[str, Any]] = []
    query_errors: list[str] = []
    forbidden: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    projection_repeats: Counter[str] = Counter()
    rows_by_query_id: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path in manifests:
        rows = read_jsonl(path)
        local_ids = Counter(str(row.get("query_id")) for row in rows)
        duplicate_ids.extend(
            f"{path.name}:{query_id}"
            for query_id, count in local_ids.items()
            if count > 1
        )
        for index, row in enumerate(rows, start=1):
            query_rows.append(row)
            query_id = str(row.get("query_id"))
            rows_by_query_id.setdefault(query_id, []).append((path.name, row))
            for hit in forbidden_key_hits(row):
                forbidden.append({"manifest": str(path.relative_to(root)), "row": index, **hit})
            try:
                query_errors.extend(
                    f"{path.name}:{index}: {message}"
                    for message in validate_query_record(
                        QueryRecord.from_dict(row), set(items), items=items
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                query_errors.append(f"{path.name}:{index}: parse error: {exc}")
    counts = Counter(str(row.get("query_scope")) for row in query_rows)
    projection_conflicts: list[str] = []
    # The canonical registry stores one query with non-exclusive roles, while
    # each manifest is a task projection.  Readiness/training gates are
    # therefore projection-local: the exact projection may be READY while the
    # semantic/stable/localized projections of the same canonical caption are
    # HOLD or EVAL_ONLY.  Compare all canonical evidence, but do not treat
    # projection metadata as a canonical-record conflict.
    projection_only_keys = {
        "view_scope",
        "view_status",
        "training_enabled",
        "candidate_training_enabled",
        "training_gate",
    }
    for query_id, records in rows_by_query_id.items():
        if len(records) < 2:
            continue
        projection_repeats[query_id] = len(records)
        canonical_rows = {
            json.dumps(
                {key: value for key, value in row.items() if key not in projection_only_keys},
                ensure_ascii=False,
                sort_keys=True,
            )
            for _, row in records
        }
        if len(canonical_rows) > 1:
            projection_conflicts.append(query_id)
    duplicate_ids = sorted(set(duplicate_ids))
    projection_conflicts.sort()
    decode_failures = _decode_sample(physical_rows, args.decode_sample)
    checksums = verify_sha256sums(root)
    result = {
        "schema_version": "qcpr-release-contract-validation-v2",
        "release": str(root),
        "physical_item_count": len(physical_rows),
        "frame_count": sum(len(row.get("frames", [])) for row in physical_rows),
        "manifest_count": len(manifests),
        "query_count": len(query_rows),
        "query_scope_counts": dict(sorted(counts.items())),
        "physical_errors": physical_errors,
        "query_errors": query_errors,
        "duplicate_query_ids": duplicate_ids,
        "projection_repeat_query_count": len(projection_repeats),
        "projection_conflicts": projection_conflicts,
        "decode_sample_limit": args.decode_sample,
        "decode_failures": decode_failures,
        "mask_free_forbidden_key_hits": forbidden,
        "sha256sums": checksums,
        "passed": not physical_errors and not query_errors and not duplicate_ids and not projection_conflicts and not decode_failures and not forbidden and checksums.get("passed", False),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
