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
    output = args.output or root / "audits/validate_qcpr_release_contract.json"
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
    for path in manifests:
        rows = read_jsonl(path)
        for index, row in enumerate(rows, start=1):
            query_rows.append(row)
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
    ids = Counter(str(row.get("query_id")) for row in query_rows)
    duplicate_ids = sorted(query_id for query_id, count in ids.items() if count > 1)
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
        "decode_sample_limit": args.decode_sample,
        "decode_failures": decode_failures,
        "mask_free_forbidden_key_hits": forbidden,
        "sha256sums": checksums,
        "passed": not physical_errors and not query_errors and not duplicate_ids and not decode_failures and not forbidden and checksums.get("passed", False),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
