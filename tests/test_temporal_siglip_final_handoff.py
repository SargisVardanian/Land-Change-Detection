from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_temporal_siglip_final_handoff import (
    HandoffError,
    validate_handoff,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_handoff(root: Path, *, forbidden: bool = False) -> Path:
    rows = [
        {"canonical_pair_id": "pair-1", "query_id": "query-1", "text": "change"},
        {"canonical_pair_id": "pair-2", "query_id": "query-2", "text": "change"},
    ]
    if forbidden:
        rows[0]["mask_path"] = "labels/mask.png"
    manifests = {}
    for name in ("train", "development", "test"):
        path = root / f"{name}.jsonl"
        _write_manifest(path, rows)
        manifests[name] = path
    handoff = {
        "authoritative_release_sha": "release-sha",
        "final_exact_train_manifest": str(manifests["train"]),
        "final_exact_development_manifest": str(manifests["development"]),
        "final_exact_test_manifest": str(manifests["test"]),
        "train_manifest_sha256": _sha256(manifests["train"]),
        "development_manifest_sha256": _sha256(manifests["development"]),
        "test_manifest_sha256": _sha256(manifests["test"]),
        "N_TRAIN_UNIQUE_PHYSICAL_PAIRS": 2,
        "BITEMPORAL_EXACT_READY": True,
        "SEMANTIC_EVAL_READY": True,
        "SEMANTIC_TRAIN_READY": False,
        "STABLE_EVAL_READY": False,
        "LOCALIZED_EVAL_READY": False,
        "LOCALIZED_TRAIN_READY": False,
        "DUBAI_EXTERNAL_READY": False,
    }
    handoff_path = root / "dataset_final_to_model.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    return handoff_path


def test_missing_handoff_is_a_hard_wait_state(tmp_path: Path) -> None:
    with pytest.raises(HandoffError, match="WAIT_DATASET_FINAL_HANDOFF"):
        validate_handoff(tmp_path / "missing.json", project_root=tmp_path)


def test_valid_handoff_checks_hashes_and_returns_counts(tmp_path: Path) -> None:
    handoff = _write_handoff(tmp_path)
    result = validate_handoff(handoff, project_root=tmp_path)
    assert result["status"] == "PASS"
    assert result["N_TRAIN_UNIQUE_PHYSICAL_PAIRS"] == 2
    assert result["manifests"]["final_exact_train_manifest"]["row_count"] == 2


def test_primary_manifest_masks_are_rejected(tmp_path: Path) -> None:
    handoff = _write_handoff(tmp_path, forbidden=True)
    with pytest.raises(HandoffError, match="MASK_FREE_PRIMARY_MANIFEST_VIOLATION"):
        validate_handoff(handoff, project_root=tmp_path)
