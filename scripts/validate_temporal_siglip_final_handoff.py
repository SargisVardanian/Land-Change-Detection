#!/usr/bin/env python3
"""Validate the Dataset-Agent handoff required by final TemporalSigLIP runs.

This validator is deliberately strict.  It never guesses a release or a
manifest when the Dataset Agent has not published the final handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "authoritative_release_sha",
    "final_exact_train_manifest",
    "final_exact_development_manifest",
    "final_exact_test_manifest",
    "train_manifest_sha256",
    "development_manifest_sha256",
    "test_manifest_sha256",
    "N_TRAIN_UNIQUE_PHYSICAL_PAIRS",
    "BITEMPORAL_EXACT_READY",
    "SEMANTIC_EVAL_READY",
    "SEMANTIC_TRAIN_READY",
    "STABLE_EVAL_READY",
    "LOCALIZED_EVAL_READY",
    "LOCALIZED_TRAIN_READY",
    "DUBAI_EXTERNAL_READY",
)

PRIMARY_MANIFESTS = (
    "final_exact_train_manifest",
    "final_exact_development_manifest",
    "final_exact_test_manifest",
)

FORBIDDEN_PRIMARY_KEYS = {
    "mask_path",
    "dense_label_path",
    "official_label",
    "semantic_map",
    "dense_label",
    "mask",
}


class HandoffError(RuntimeError):
    """Raised when a final Dataset-Agent contract is not safe to consume."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffError(f"INVALID_HANDOFF_JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HandoffError("INVALID_HANDOFF_ROOT: expected a JSON object")
    return value


def _field(handoff: dict[str, Any], name: str) -> Any:
    if name in handoff:
        return handoff[name]
    capabilities = handoff.get("capabilities")
    if isinstance(capabilities, dict) and name in capabilities:
        return capabilities[name]
    return None


def _resolve_path(raw: Any, *, handoff_path: Path, project_root: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise HandoffError("INVALID_HANDOFF_PATH: expected a non-empty string")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_handoff = (handoff_path.parent / candidate).resolve()
    if from_handoff.exists():
        return from_handoff
    return (project_root / candidate).resolve()


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_PRIMARY_KEYS:
                violations.append(f"{path}.{key_text}")
            violations.extend(_walk_forbidden(child, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return violations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HandoffError(f"MANIFEST_READ_ERROR: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HandoffError(f"INVALID_MANIFEST_JSON: {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise HandoffError(f"INVALID_MANIFEST_ROW: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise HandoffError(f"EMPTY_PRIMARY_MANIFEST: {path}")
    return rows


def _id_sha(rows: list[dict[str, Any]], key: str) -> str | None:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            return None
        values.append(str(value))
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_handoff(
    handoff_path: Path,
    *,
    project_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not handoff_path.is_file():
        result = {
            "status": "WAIT_DATASET_FINAL_HANDOFF",
            "handoff": str(handoff_path),
            "missing": True,
            "message": "Final Dataset-Agent handoff is absent; no release or budget may be guessed.",
        }
        if output_path is not None:
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise HandoffError("WAIT_DATASET_FINAL_HANDOFF")

    handoff = _load_json(handoff_path)
    missing = [field for field in REQUIRED_FIELDS if _field(handoff, field) is None]
    if missing:
        raise HandoffError("MISSING_HANDOFF_FIELDS: " + ",".join(missing))
    if _field(handoff, "BITEMPORAL_EXACT_READY") is not True:
        raise HandoffError("BITEMPORAL_EXACT_NOT_READY")
    pair_count = _field(handoff, "N_TRAIN_UNIQUE_PHYSICAL_PAIRS")
    if not isinstance(pair_count, int) or pair_count <= 0:
        raise HandoffError("INVALID_N_TRAIN_UNIQUE_PHYSICAL_PAIRS")

    manifest_results: dict[str, Any] = {}
    for path_field in PRIMARY_MANIFESTS:
        manifest_path = _resolve_path(
            _field(handoff, path_field), handoff_path=handoff_path, project_root=project_root
        )
        if not manifest_path.is_file():
            raise HandoffError(f"MISSING_PRIMARY_MANIFEST: {manifest_path}")
        expected_hash_field = {
            "final_exact_train_manifest": "train_manifest_sha256",
            "final_exact_development_manifest": "development_manifest_sha256",
            "final_exact_test_manifest": "test_manifest_sha256",
        }[path_field]
        expected_hash = _field(handoff, expected_hash_field)
        actual_hash = _sha256(manifest_path)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise HandoffError(f"MANIFEST_SHA256_MISMATCH: {path_field}")
        rows = _read_jsonl(manifest_path)
        violations = _walk_forbidden(rows)
        if violations:
            raise HandoffError(
                f"MASK_FREE_PRIMARY_MANIFEST_VIOLATION: {path_field}: {violations[:5]}"
            )
        manifest_results[path_field] = {
            "path": str(manifest_path),
            "sha256": actual_hash,
            "row_count": len(rows),
            "ordered_pair_id_sha256": _id_sha(rows, "canonical_pair_id"),
            "ordered_query_id_sha256": _id_sha(rows, "query_id"),
        }

    result = {
        "status": "PASS",
        "handoff": str(handoff_path),
        "authoritative_release_sha": _field(handoff, "authoritative_release_sha"),
        "N_TRAIN_UNIQUE_PHYSICAL_PAIRS": pair_count,
        "capabilities": {
            field: _field(handoff, field)
            for field in REQUIRED_FIELDS
            if field.endswith("_READY")
        },
        "manifests": manifest_results,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_handoff(
            args.handoff,
            project_root=args.project_root,
            output_path=args.output,
        )
    except HandoffError as exc:
        print(json.dumps({"status": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
