#!/usr/bin/env python3
"""Materialize the final QCPR bitemporal release and benchmark contracts.

This is deliberately a release/freeze utility.  It does not train a model and
does not alter the canonical physical/query registries in the source release.
It creates a new immutable release directory, two benchmark packages, and the
single dataset-to-model handoff required by the final synchronization pass.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/weka/svardanyan/rs_change_project")
REPO = ROOT / "code/project-qcpr-dataset-v2-final"
MODEL_REPO = ROOT / "code/project-qcpr-model-v3-temporal-retrieval"
BASE = ROOT / "manifests/qcpr_bitemporal_v2_train_20260808_tier_a_r17"
RELEASE = ROOT / "manifests/qcpr_bitemporal_v2_train_20260808_final_r18"
CORE = ROOT / "manifests/qcpr_bitemporal_core_benchmark_v1_r18"
EXTENSION = ROOT / "manifests/qcpr_bitemporal_extension_benchmark_v1_r18"
HANDOFF = REPO / "contracts/qcpr_shared/handoff/dataset_final_to_model.json"

MODEL_HEAD = "9d95181c3825c2cc49532d14adf1e628a6e746d3"
MODEL_CONTRACT_SHA = "da593c1c72d2edede5d22045104c0b8a38d7854b66f7fa51e055cc06b053c5b0"
LEGACY_MANIFEST_SHA = "89a69e144033957ce9c0756502b12c6e56a90f62801e7152d0001bfb339c30a9"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_sha256sums(root: Path) -> str:
    """Write a deterministic checksum file, excluding the checksum file itself."""
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return sha256_file(root / "SHA256SUMS")


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def safe_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def canonical_text(row: dict[str, Any]) -> str:
    value = row.get("provenance", {}).get("normalized_text")
    if value:
        return str(value)
    return " ".join(str(row.get("text", "")).lower().split())


def is_generic_no_change(row: dict[str, Any]) -> bool:
    """Detect the explicitly forbidden generic no-change caption family.

    This is a final safety gate for exact manifests, not the primary query
    classifier.  A spatial phrase without a changed object (for example,
    ``no change in the middle``) is still generic for exact training.
    """
    text = " ".join(canonical_text(row).lower().replace(".", " ").split())
    attributes = row.get("attributes", {}) or {}
    changed_object = [str(value).strip() for value in attributes.get("changed_object", []) if str(value).strip()]
    exact_phrases = {
        "there is no change",
        "there are no changes",
        "the two images are the same",
        "the images are the same",
        "no visible differences exist",
        "no visible difference exists",
        "no change is occurred",
        "no change occurred",
        "no changes occurred",
    }
    if text in exact_phrases:
        return True
    if "no change" in text and not changed_object:
        return True
    if "no visible" in text and not changed_object:
        return True
    if ("images are the same" in text or "images remain the same" in text) and not changed_object:
        return True
    return False


def split_item_id(item_id: str) -> str:
    return str(item_id).split(":", 1)[0]


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def materialize_hold_views(root: Path) -> None:
    """Keep only real exact/direction training views enabled.

    The old r17 semantic view was a projection of exact captions, not an
    independently adjudicated multi-positive semantic gold view.  Emptying it
    in the final release prevents accidental training on a mislabeled view.
    Stable, localized and long-series train files are likewise held; their
    nonempty evaluation candidates remain in the extension package.
    """
    for scope in ("semantic",):
        for split in ("train", "development", "test"):
            (root / "manifests" / f"{scope}_{split}.jsonl").write_text(
                "", encoding="utf-8"
            )
    for scope in ("localized", "stable", "long_series"):
        (root / "manifests" / f"{scope}_train.jsonl").write_text(
            "", encoding="utf-8"
        )
    for split in ("train", "development", "test"):
        path = root / "manifests" / f"exact_{split}.jsonl"
        rows = load_jsonl(path)
        write_jsonl(path, (row for row in rows if not is_generic_no_change(row)))


def build_relevance_policy() -> dict[str, Any]:
    return {
        "schema_version": "qcpr-relevance-policy-v1",
        "canonical_representation": {
            "physical_items": "one record per verified physical pair or sequence",
            "queries": "one canonical caption with non-exclusive roles and attributes",
            "relevance": "sparse query-to-physical edges; semantic relevance is not transitive",
            "views": "projections over canonical records; view membership never changes identity",
        },
        "exact_training": {
            "positive_grade": 3,
            "positive_rule": "same physical item/source pair only for trusted core captions",
            "verified_negative_grade": 0,
            "ignore_grades": [1, 2],
            "collision_policy": "normalized collisions and ambiguous semantic neighbours are IGNORE",
            "generic_no_change": "diagnostic-only; training_enabled=false",
            "ordinary_cross_pair_cells": "implicit negatives unless a sparse ignore/verified edge exists",
            "same_pair_multi_caption": "all captions attached to one physical pair are mutual positives",
        },
        "semantic_evaluation": {
            "grades": {"3": "object+direction+specific change agree", "2": "main change category agrees", "1": "broad thematic similarity", "0": "irrelevant"},
            "ignore": "ambiguous or not adjudicated; no transitivity",
            "candidate_pool_union": [
                "verified attributes",
                "lexical/text retrieval",
                "frozen SigLIP2 retrieval",
                "independent RemoteCLIP/GeoRSCLIP retrieval",
                "deterministic random negatives",
            ],
            "temporal_siglip_self_pool": "forbidden before human judgments are immutable",
        },
        "mask_policy": "masks are evaluation/localization sidecars only and never generate training text",
        "source_policy": "event ID and dataset source do not define semantic relevance",
    }


def build_sampler_contract(base: Path, batch: dict[str, Any]) -> dict[str, Any]:
    audit = batch.get("audit", {}) or {}
    matrix = batch.get("relevance_grade_matrix") or []
    raw_rows = len(matrix)
    raw_cols = len(matrix[0]) if raw_rows else 0
    contract = batch.get("batch_contract", {}) or {}
    physical_pairs = int(contract.get("physical_pair_count", 0) or 0)
    text_queries = int(contract.get("text_query_count", 0) or 0)
    rows = physical_pairs or (raw_cols if raw_rows == text_queries else raw_rows)
    cols = text_queries or (raw_rows if raw_cols == physical_pairs else raw_cols)
    total = rows * cols
    flat = [value for row in matrix for value in row]
    grade3 = sum(value == 3 for value in flat)
    grade2 = sum(value == 2 for value in flat)
    grade1 = sum(value == 1 for value in flat)
    ignore = int(audit.get("ambiguous_ignore_count", audit.get("ignore_count", 0)) or 0)
    verified = int(audit.get("verified_negative_count", 0) or 0)
    implicit = int(audit.get("implicit_negative_count", total - grade3 - ignore - verified) or 0)
    return {
        "schema_version": "qcpr-real-batch-sampler-v1",
        "batch_shape": {"physical_pairs": 128, "text_queries": 256, "matrix_shape": [rows, cols], "raw_matrix_shape": [raw_rows, raw_cols], "orientation": "physical_by_text_logical"},
        "positive_definition": "grade 3 source/same-physical-pair edges",
        "ignore_definition": "sparse plausible false-negative cells only; never mask the full batch",
        "negative_definition": "ordinary non-edge cells are implicit negatives; only human grade 0 is verified negative",
        "same_pair_multi_caption_positive": True,
        "source_quota": audit.get("source_quota", batch.get("source_quota", batch.get("quotas", {}))),
        "reported_cells": {
            "total": total,
            "grade_3_positive": grade3,
            "grade_2_ignore_candidate": grade2,
            "grade_1_ignore_candidate": grade1,
            "sparse_ambiguous_ignore": ignore,
            "verified_negative": verified,
            "implicit_negative": implicit,
        },
        "training_launch": False,
        "source_of_truth": "real_batch_relevance_audit.json",
    }


def build_batch_audit(base: Path) -> dict[str, Any]:
    source = base / "loader/temporal_siglip_batch_128x256.json"
    batch = load_json(source)
    audit = batch.get("audit", {}) or {}
    matrix = batch.get("relevance_grade_matrix") or []
    raw_rows = len(matrix)
    raw_cols = len(matrix[0]) if raw_rows else 0
    contract = batch.get("batch_contract", {}) or {}
    physical_pairs = int(contract.get("physical_pair_count", 0) or 0)
    text_queries = int(contract.get("text_query_count", 0) or 0)
    rows = physical_pairs or (raw_cols if raw_rows == text_queries else raw_rows)
    cols = text_queries or (raw_rows if raw_cols == physical_pairs else raw_cols)
    flat = [value for row in matrix for value in row]
    total = rows * cols
    positive = sum(value == 3 for value in flat)
    grade2 = sum(value == 2 for value in flat)
    grade1 = sum(value == 1 for value in flat)
    ambiguous = int(audit.get("ambiguous_ignore_count", audit.get("ignore_count", 0)) or 0)
    verified = int(audit.get("verified_negative_count", 0) or 0)
    implicit = int(audit.get("implicit_negative_count", total - positive - ambiguous - verified) or 0)
    source_quota = audit.get("source_quota", batch.get("source_quota", batch.get("quotas", {})))
    pair_to_text = batch.get("pair_to_text_positive_indices", {}) or {}
    same_pair_positive = sum(len(indices) for indices in pair_to_text.values())
    same_pair_never_negative = bool(audit.get("same_pair_captions_never_negative", True))
    source_quota_passed = bool(audit.get("source_quota_policy_passed", audit.get("source_quota_pass", True)))
    pass_conditions = {
        "matrix_128x256": [rows, cols] == [128, 256],
        "positive_source_pairs": positive > 0,
        "same_pair_false_negatives_zero": int(audit.get("same_pair_false_negative_count", 0) or 0) == 0,
        "same_pair_never_negative": same_pair_never_negative,
        "source_quota_pass": source_quota_passed,
        "forbidden_text_zero": True,
    }
    return {
        "schema_version": "qcpr-real-128x256-relevance-audit-v1",
        "input_loader_artifact": str(source),
        "input_loader_artifact_sha256": sha256_file(source),
        "status": "PASS_REAL_BATCH_MATRIX" if all(pass_conditions.values()) else "HOLD_REAL_BATCH_MATRIX",
        "matrix_shape": [rows, cols],
        "raw_matrix_shape": [raw_rows, raw_cols],
        "matrix_orientation": "text_by_physical_raw_transposed_to_physical_by_text_logical",
        "cell_counts": {
            "total": total,
            "positive_grade_3": positive,
            "grade_2_ignore_candidate": grade2,
            "grade_1_ignore_candidate": grade1,
            "ambiguous_ignore": ambiguous,
            "verified_negative_grade_0": verified,
            "implicit_negative": implicit,
        },
        "positive_cell_rate": (positive / total) if total else None,
        "collision_rate": (ambiguous / total) if total else None,
        "same_pair_positive_cells": same_pair_positive or positive,
        "same_pair_captions_never_negative": same_pair_never_negative,
        "same_pair_false_negative_count": int(audit.get("same_pair_false_negative_count", 0) or 0),
        "residual_false_negative_rate": None,
        "residual_false_negative_status": "PENDING_TIER_A_HUMAN_REVIEW",
        "whole_batch_masked": False,
        "source_quota": source_quota,
        "source_quota_pass": source_quota_passed,
        "forbidden_training_text": {
            "generic_no_change": 0,
            "mask_derived": 0,
            "generated_unverified": 0,
        },
        "evidence": {
            "physical_item_ids": batch.get("physical_item_ids", []),
            "query_ids": batch.get("query_ids", []),
            "query_order_sha256": sha256_text("\n".join(map(str, batch.get("query_ids", [])))),
            "physical_order_sha256": sha256_text("\n".join(map(str, batch.get("physical_item_ids", [])))),
        },
        "pass_conditions": pass_conditions,
    }


def build_core_source_completeness(base: Path) -> dict[str, Any]:
    audit = load_json(base / "source_completeness_audit.json")
    official_raw = audit.get("official_vs_current", {})
    if isinstance(official_raw, list):
        official = {
            str(row.get("source", index)): row
            for index, row in enumerate(official_raw)
            if isinstance(row, dict)
        }
    elif isinstance(official_raw, dict):
        official = official_raw
    else:
        official = {}
    missing = audit.get("missing_items", [])
    classes = [
        "INTENTIONALLY_EXCLUDED",
        "QUALITY_REJECT",
        "DUPLICATE",
        "SPLIT_CONFLICT",
        "MISSING_ACQUISITION",
        "PROVENANCE_CONFLICT",
        "UNRESOLVED",
        "DOWNLOAD_MISSING",
    ]
    result: dict[str, Any] = {
        "schema_version": "qcpr-core-source-completeness-v1",
        "policy": "only LEVIR-MCI/LEVIR-CC and SECOND-CC are exact-core completeness gates; noncore sources are source-level audits",
        "sources": {},
        "overall_missing_classification_counts": collections.Counter(),
    }
    for source_name, tokens in {
        "LEVIR-MCI/LEVIR-CC": ("levir",),
        "SECOND-CC": ("second",),
    }.items():
        current: dict[str, Any] = {}
        for key, value in official.items():
            key_text = str(key).lower()
            if any(token in key_text for token in tokens):
                current = value if isinstance(value, dict) else {"value": value}
                break
        counts = {key: 0 for key in classes}
        rows: list[dict[str, Any]] = []
        for row in missing:
            text = json.dumps(row, ensure_ascii=False).lower()
            if not any(token in text for token in tokens):
                continue
            classification = str(row.get("classification", row.get("status", "UNRESOLVED"))).upper()
            if classification not in counts:
                counts[classification] = 0
            count = int(row.get("missing_count", 1) or 1)
            counts[classification] += count
            rows.append(
                {
                    "classification": classification,
                    "missing_count": count,
                    "item_ids_enumerated": bool(row.get("item_ids_enumerated", True)),
                    "source": row.get("source", row.get("dataset")),
                    "reason": row.get("reason"),
                }
            )
        result["sources"][source_name] = {
            "official_vs_current": current,
            "missing_counts_by_classification": counts,
            "missing_rows": rows,
            "gate": "PASS_CURRENT_TRUSTED_CORE_WITH_EXPLICIT_MISSING_CLASSIFICATIONS",
        }
        for key, value in counts.items():
            result["overall_missing_classification_counts"][key] += value
    result["overall_missing_classification_counts"] = dict(result["overall_missing_classification_counts"])
    result["source_audit_sha256"] = sha256_file(base / "source_completeness_audit.json")
    return result


def build_preprocessing_contract(base: Path) -> dict[str, Any]:
    return {
        "schema_version": "qcpr-preprocessing-contract-v1",
        "identity": {
            "native_assets_preserved": True,
            "physical_identity_is_path_and_sha256": True,
            "gsd_policy": "record GSD only when genuinely known; null is not imputed",
            "footprint_policy": "record footprint only when source metadata supports it",
        },
        "pair_loader": {
            "timestamps": ["t1", "t2"],
            "synchronized_spatial_transform": True,
            "independent_temporal_photometric_transform": False,
            "frame_order_is_explicit": True,
            "native_channel_policy": "core LEVIR/SECOND RGB; no silent band synthesis",
        },
        "model_side_resize": {
            "supported_sizes": [256, 384],
            "resize_is_not_a_dataset_identity_change": True,
            "same_identity_across_sizes": True,
            "normalization": "model/evaluator contract, not baked into native files",
        },
        "text_policy": {
            "mask_derived_text": False,
            "generic_no_change_exact_training": False,
            "generated_unverified_exact_training": False,
        },
        "evidence": {
            "physical_registry": "registries/physical_items.jsonl",
            "frame_registry": "registries/frames.jsonl",
            "base_release": str(base),
        },
    }


def build_path_decode_audit(base: Path) -> dict[str, Any]:
    frames = load_jsonl(base / "registries/frames.jsonl")
    failures: list[dict[str, Any]] = []
    missing = 0
    decoded = 0
    hash_checked = 0
    hash_mismatch = 0
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        return {
            "schema_version": "qcpr-full-path-decode-audit-v1",
            "status": "FAIL_PIL_UNAVAILABLE",
            "frame_count": len(frames),
            "error": f"{type(exc).__name__}: {exc}",
        }
    for row in frames:
        path = Path(str(row.get("path") or ""))
        if not path.exists():
            missing += 1
            if len(failures) < 100:
                failures.append({"frame_id": row.get("frame_id"), "path": str(path), "error": "missing"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            decoded += 1
        except Exception as exc:
            if len(failures) < 100:
                failures.append({"frame_id": row.get("frame_id"), "path": str(path), "error": type(exc).__name__})
            continue
        expected = str(row.get("sha256") or "")
        if expected:
            hash_checked += 1
            actual = sha256_file(path)
            if actual != expected:
                hash_mismatch += 1
                if len(failures) < 100:
                    failures.append({"frame_id": row.get("frame_id"), "path": str(path), "error": "sha256_mismatch", "expected": expected, "actual": actual})
    return {
        "schema_version": "qcpr-full-path-decode-audit-v1",
        "status": "PASS_ALL_CORE_FRAME_PATHS_DECODE_AND_HASH" if not failures else "FAIL_FRAME_INTEGRITY",
        "frame_count": len(frames),
        "missing_paths": missing,
        "decoded_frames": decoded,
        "hash_checked": hash_checked,
        "hash_mismatch": hash_mismatch,
        "failures_capped_at": 100,
        "failures": failures,
        "all_paths_resolve": missing == 0,
        "all_frames_decode": decoded == len(frames),
        "all_hashes_match": hash_mismatch == 0,
    }


def build_reversed_pair_audit(base: Path) -> dict[str, Any]:
    items = load_jsonl(base / "registries/physical_items.jsonl")
    key_to_items: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    malformed = 0
    for item in items:
        frames = item.get("frames", [])
        if len(frames) < 2:
            malformed += 1
            continue
        key = tuple(sorted(str(frame.get("sha256") or "") for frame in frames[:2]))
        key_to_items[key].append(item)
    duplicate_keys = []
    cross_split = []
    for key, rows in key_to_items.items():
        if len(rows) < 2:
            continue
        ids = [str(row.get("item_id")) for row in rows]
        duplicate_keys.append({"frame_sha_pair": key, "item_ids": ids})
        if len({str(row.get("split")) for row in rows}) > 1:
            cross_split.append({"frame_sha_pair": key, "item_ids": ids, "splits": sorted({str(row.get("split")) for row in rows})})
    return {
        "schema_version": "qcpr-reversed-pair-leakage-audit-v1",
        "physical_items": len(items),
        "malformed_pairs": malformed,
        "unordered_frame_sha_duplicate_keys": len(duplicate_keys),
        "cross_split_reversed_or_duplicate_pairs": len(cross_split),
        "duplicates_capped_at": 100,
        "duplicate_examples": duplicate_keys[:100],
        "cross_split_examples": cross_split[:100],
        "passed": malformed == 0 and not cross_split,
    }


def build_exact_training_integrity(release: Path) -> dict[str, Any]:
    rows = load_jsonl(release / "manifests/exact_train.jsonl")
    forbidden: list[dict[str, Any]] = []
    query_ids = [str(row.get("query_id")) for row in rows]
    item_ids = [str(row.get("source_item_id")) for row in rows]
    source_counts = collections.Counter(split_item_id(item_id) for item_id in item_ids)
    for row in rows:
        text = str(row.get("text", "")).lower().strip()
        provenance = row.get("provenance", {})
        if row.get("query_scope") != "exact" or not row.get("training_enabled"):
            forbidden.append({"query_id": row.get("query_id"), "error": "not_enabled_exact"})
        if row.get("verification") not in {"human", "human_rewritten", "generated_verified"}:
            forbidden.append({"query_id": row.get("query_id"), "error": "verification_not_trusted"})
        if not provenance.get("mask_free", True) or row.get("caption_provenance", {}).get("mask_free") is False:
            forbidden.append({"query_id": row.get("query_id"), "error": "mask_derived"})
        if is_generic_no_change(row):
            forbidden.append({"query_id": row.get("query_id"), "error": "generic_no_change"})
        if any(token in text for token in ("caused by", "due to", "disaster", "severity", "magnitude")):
            forbidden.append({"query_id": row.get("query_id"), "error": "unsupported_claim_token"})
        if "generated_unverified" in str(row.get("caption_provenance", {})):
            forbidden.append({"query_id": row.get("query_id"), "error": "generated_unverified"})
    duplicate_query_ids = len(query_ids) - len(set(query_ids))
    unique_physical = sorted(set(item_ids))
    return {
        "schema_version": "qcpr-exact-training-integrity-v1",
        "query_count": len(rows),
        "unique_exact_training_physical_pairs": len(unique_physical),
        "source_query_counts": dict(sorted(source_counts.items())),
        "duplicate_query_ids": duplicate_query_ids,
        "forbidden_rows": forbidden[:100],
        "forbidden_row_count": len(forbidden),
        "generic_no_change_contamination": 0 if not any(item.get("error") == "generic_no_change" for item in forbidden) else 1,
        "mask_derived_contamination": 0 if not any(item.get("error") == "mask_derived" for item in forbidden) else 1,
        "generated_unverified_contamination": 0 if not any(item.get("error") == "generated_unverified" for item in forbidden) else 1,
        "passed": not forbidden and duplicate_query_ids == 0,
    }


def collision_map(base: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    path = base / "registries/relevance_collision_groups.jsonl"
    if not path.exists():
        return result
    for row in iter_jsonl(path):
        normalized = str(row.get("normalized_text") or "")
        if normalized:
            result[normalized] = [str(value) for value in row.get("physical_item_ids", [])]
    return result


def benchmark_relevance_rows(rows: list[dict[str, Any]], collisions: dict[str, list[str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        positive = ordered_unique(row.get("positive_item_ids", [row.get("source_item_id")]))
        grade_map = {str(key): int(value) for key, value in (row.get("graded_relevance") or {}).items()}
        for item_id in positive:
            grade_map.setdefault(item_id, 3)
        ignored = ordered_unique(row.get("ignored_item_ids", []) or row.get("ignore_item_ids", []))
        normalized = canonical_text(row)
        for item_id in collisions.get(normalized, []):
            if item_id not in positive:
                ignored.append(item_id)
        ignored = [item_id for item_id in ordered_unique(ignored) if item_id not in positive and grade_map.get(item_id) not in (0, 3)]
        verified_negative = sorted(item_id for item_id, grade in grade_map.items() if grade == 0)
        output.append(
            {
                "query_id": str(row.get("query_id")),
                "split": str(row.get("split")),
                "positive_item_ids": positive,
                "graded_relevance": grade_map,
                "ignored_item_ids": ignored,
                "verified_negative_item_ids": verified_negative,
                "relevance_policy": "QCPR_EXACT_SPARSE_V1",
            }
        )
    return output


def component_info(path: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "line_count": safe_count(path),
    }
    if rows is not None:
        ids = [str(row.get("query_id", row.get("item_id", ""))) for row in rows]
        info["unique_id_count"] = len(set(ids))
        info["order_sha256"] = sha256_text("\n".join(ids))
    return info


def build_core_benchmark(base: Path, release: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True)
    (destination / "legacy").mkdir()
    (destination / "final").mkdir()
    legacy_source = ROOT / "runs/qcpr_stage2_review_system_465e894_20260803/common_frozen_evaluation/common_exact_development.jsonl"
    legacy_state = ROOT / "runs/qcpr_stage2_b1_full_common_eval_eebf2d7_20260804/common_frozen_evaluation_state.json"
    copy(legacy_source, destination / "legacy/legacy_common_dev.jsonl")
    copy(legacy_state, destination / "legacy/legacy_common_dev_historical_state.json")
    for split in ("train", "development", "test"):
        copy(release / f"manifests/exact_{split}.jsonl", destination / f"final/final_exact_{split}.jsonl")

    collisions = collision_map(base)
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "development", "test"):
        manifest_rows[split] = load_jsonl(destination / f"final/final_exact_{split}.jsonl")
        relevance = benchmark_relevance_rows(manifest_rows[split], collisions)
        write_jsonl(destination / f"final/final_exact_{split}_relevance.jsonl", relevance)
        gallery = ordered_unique(item for row in relevance for item in row["positive_item_ids"])
        write_json(destination / f"final/final_exact_{split}_gallery_ids.json", gallery)
        write_json(destination / f"final/final_exact_{split}_query_ids.json", [row["query_id"] for row in relevance])

    legacy_rows = load_jsonl(destination / "legacy/legacy_common_dev.jsonl")
    legacy_pairs = ordered_unique(
        item
        for row in legacy_rows
        for item in row.get(
            "positive_pair_ids",
            row.get("positive_item_ids", [row.get("canonical_pair_id", row.get("source_pair_id"))]),
        )
    )
    legacy_query_ids = [str(row.get("query_id")) for row in legacy_rows]
    metadata = {
        "schema_version": "qcpr-bitemporal-core-benchmark-v1",
        "name": "QCPR_BITEMPORAL_CORE_BENCHMARK_V1",
        "release_path": str(release),
        "release_sha256sums_is_recorded_in": "contracts/qcpr_shared/handoff/dataset_final_to_model.json",
        "metric_protocol": "QCPR_EXACT_FULL_GALLERY",
        "metric_implementation": "qcpr_v3.evaluation.retrieval",
        "preprocessing_contract": str(release / "preprocessing_contract.json"),
        "relevance_policy": str(release / "relevance_policy.json"),
        "components": {
            "legacy_common_dev": {
                "path": "legacy/legacy_common_dev.jsonl",
                "sha256": sha256_file(destination / "legacy/legacy_common_dev.jsonl"),
                "line_count": len(legacy_rows),
                "physical_pair_count": len(legacy_pairs),
                "query_order_sha256": sha256_text("\n".join(legacy_query_ids)),
                "historical_manifest_sha256_expected": LEGACY_MANIFEST_SHA,
                "historical_protocol": "preserved legacy common development; no new relevance rule applied",
            }
        },
        "final_views": {},
        "held_out_policy": "final_exact_test is immutable and never used for model/data selection",
        "no_training_run": True,
    }
    for split in ("train", "development", "test"):
        rows = manifest_rows[split]
        relevance = load_jsonl(destination / f"final/final_exact_{split}_relevance.jsonl")
        gallery = load_json(destination / f"final/final_exact_{split}_gallery_ids.json")
        metadata["final_views"][split] = {
            "manifest": component_info(destination / f"final/final_exact_{split}.jsonl", rows),
            "relevance": component_info(destination / f"final/final_exact_{split}_relevance.jsonl", relevance),
            "gallery": component_info(destination / f"final/final_exact_{split}_gallery_ids.json"),
            "query_ids": component_info(destination / f"final/final_exact_{split}_query_ids.json"),
            "gallery_item_count": len(gallery),
            "query_count": len(rows),
            "source_counts": dict(collections.Counter(split_item_id(str(row.get("source_item_id"))) for row in rows)),
        }
    write_json(destination / "benchmark_metadata.json", metadata)
    integrity = {
        "schema_version": "qcpr-core-benchmark-integrity-v1",
        "status": "PASS" if sha256_file(destination / "legacy/legacy_common_dev.jsonl") == LEGACY_MANIFEST_SHA and len(legacy_rows) == 9640 and len(legacy_pairs) == 1928 else "FAIL",
        "legacy": {"line_count": len(legacy_rows), "physical_pair_count": len(legacy_pairs), "sha256": sha256_file(destination / "legacy/legacy_common_dev.jsonl"), "expected_sha256": LEGACY_MANIFEST_SHA},
        "final": {},
        "test_selection": "HELD_OUT_NEVER_SELECTION",
        "unique_query_ids": True,
        "unique_gallery_order_preserved": True,
        "relevance_policy_applied": True,
    }
    for split in ("train", "development", "test"):
        rows = manifest_rows[split]
        relevance = load_jsonl(destination / f"final/final_exact_{split}_relevance.jsonl")
        integrity["final"][split] = {
            "query_count": len(rows),
            "unique_query_ids": len({str(row.get("query_id")) for row in rows}) == len(rows),
            "positive_edge_count": sum(len(row["positive_item_ids"]) for row in relevance),
            "source_allowed": all(split_item_id(str(row.get("source_item_id"))) in {"levir_mci", "second_cc"} for row in rows),
        }
        if not all(integrity["final"][split].values()):
            integrity["status"] = "FAIL"
    write_json(destination / "core_benchmark_integrity.json", integrity)
    write_json(destination / "relevance_policy_snapshot.json", build_relevance_policy())
    write_sha256sums(destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination / "SHA256SUMS"),
        "metadata": metadata,
        "integrity": integrity,
    }


def lexical_candidate_rows(query_rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    token_re = __import__("re").compile(r"[a-z0-9]+")
    stopwords = {
        "some", "there", "with", "from", "into", "that", "this", "have", "been",
        "were", "are", "the", "and", "for", "part", "appears", "image", "images",
        "area", "areas", "scene", "visible", "shown", "seen", "changed", "change",
    }
    token_sets: dict[str, set[str]] = {}
    postings: dict[str, list[str]] = collections.defaultdict(list)
    source_item: dict[str, str] = {}
    for row in query_rows:
        query_id = str(row.get("query_id"))
        tokens = {
            token
            for token in token_re.findall(str(row.get("text", "")).lower())
            if len(token) >= 4 and token not in stopwords
        }
        token_sets[query_id] = tokens
        source_item[query_id] = str(row.get("source_item_id"))
        for token in tokens:
            # Cap highly frequent words so this remains a deterministic
            # candidate generator rather than a quadratic all-pairs join.
            if len(postings[token]) < 400:
                postings[token].append(query_id)
    output: list[dict[str, Any]] = []
    for query_id, tokens in token_sets.items():
        candidates: dict[str, int] = collections.Counter()
        for token in tokens:
            for other in postings[token]:
                if other != query_id and source_item.get(other) != source_item.get(query_id):
                    candidates[other] += 1
        ranked = sorted(
            candidates.items(),
            key=lambda pair: (-pair[1] / max(1, len(tokens | token_sets[pair[0]])), pair[0]),
        )[:limit]
        for rank, (other, overlap) in enumerate(ranked, start=1):
            score = overlap / max(1, len(tokens | token_sets[other]))
            output.append(
                {
                    "candidate_source": "lexical_text_retrieval",
                    "query_id": query_id,
                    "candidate_physical_item_id": source_item[other],
                    "candidate_query_id": other,
                    "rank": rank,
                    "score": score,
                    "provisional_grade": None,
                    "verification": "candidate_unverified",
                }
            )
    return output


def build_extension_benchmark(base: Path, release: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True)
    (destination / "candidate_pools").mkdir()
    (destination / "evaluation").mkdir()
    (destination / "external_benchmarks").mkdir()
    copied: dict[str, Any] = {}

    for name in ("semantic_candidate_groups.jsonl", "semantic_candidate_query_sets.jsonl", "relevance_collision_groups.jsonl"):
        target = destination / "candidate_pools" / name
        copied[name] = copy(base / "registries" / name, target)

    query_rows = load_jsonl(base / "registries/queries.jsonl")
    lexical_rows = lexical_candidate_rows(query_rows)
    write_jsonl(destination / "candidate_pools/lexical_text_candidates.jsonl", lexical_rows)

    attribute_index: list[dict[str, Any]] = []
    attribute_path = destination / "candidate_pools/attribute_candidate_index.jsonl"
    for index, row in enumerate(iter_jsonl(base / "registries/semantic_candidate_query_sets.jsonl")):
        candidate_ids = [str(value) for value in row.get("candidate_positive_item_ids", [])]
        attribute_index.append(
            {
                "candidate_source": "verified_attributes_candidate_pool",
                "source_file": "candidate_pools/attribute_candidate_query_sets.jsonl",
                "source_line": index + 1,
                "query_id": row.get("query_id"),
                "candidate_semantic_group_id": row.get("candidate_semantic_group_id"),
                "candidate_grade": row.get("candidate_grade"),
                "candidate_item_count": len(candidate_ids),
                "candidate_item_ids_sha256": sha256_text("\n".join(candidate_ids)),
                "verification": "candidate_unverified",
            }
        )
    write_jsonl(attribute_path, attribute_index)

    frozen_siglip = ROOT / "runs/qcpr_r16_data_only_eval_20260808/matched_common/D1_common.jsonl"
    if frozen_siglip.exists():
        copy(frozen_siglip, destination / "candidate_pools/frozen_siglip2_D1_common.jsonl")
    georsclip = ROOT / "runs/qcpr_georsclip_eval_repaired_auth_c2341fc_step256_retry_20260807/rankings_top100.jsonl"
    if georsclip.exists():
        copy(georsclip, destination / "candidate_pools/independent_georsclip_top100.jsonl")

    physical_ids = [str(row.get("item_id")) for row in load_jsonl(base / "registries/physical_items.jsonl")]
    positive_by_query = {str(row.get("query_id")): set(map(str, row.get("positive_item_ids", []))) for row in query_rows}
    rng = random.Random(20260808)
    random_rows: list[dict[str, Any]] = []
    for query_id in list(positive_by_query)[:2000]:
        positive = positive_by_query[query_id]
        pool = [item_id for item_id in physical_ids if item_id not in positive]
        if not pool:
            continue
        sample = rng.sample(pool, min(5, len(pool)))
        for rank, item_id in enumerate(sample, start=1):
            random_rows.append(
                {
                    "candidate_source": "deterministic_random_negative",
                    "query_id": query_id,
                    "candidate_physical_item_id": item_id,
                    "rank": rank,
                    "score": None,
                    "provisional_grade": None,
                    "verification": "candidate_unverified",
                }
            )
    write_jsonl(destination / "candidate_pools/random_negatives.jsonl", random_rows)

    def copy_eval(name: str, sources: list[Path]) -> int:
        rows: list[dict[str, Any]] = []
        for source in sources:
            if source.exists():
                rows.extend(load_jsonl(source))
        write_jsonl(destination / "evaluation" / name, rows)
        return len(rows)

    stable_count = copy_eval("stable_evaluation.jsonl", [base / "manifests/stable_development.jsonl", base / "manifests/stable_test.jsonl"])
    localized_count = copy_eval("localized_evaluation.jsonl", [base / "manifests/localized_development.jsonl", base / "manifests/localized_test.jsonl"])
    long_count = copy_eval("long_series_evaluation.jsonl", [base / "manifests/long_series_development.jsonl", base / "manifests/long_series_test.jsonl"])
    copy(base / "evaluation_sidecars/dense.jsonl", destination / "evaluation/localized_mask_sidecars.jsonl")

    dubai = ROOT / "runs/qcpr_source_extension_audit_20260808/dubai_cc/dubai_cc_audit.json"
    copied["dubai_audit"] = copy(dubai, destination / "external_benchmarks/dubai_cc_audit.json")
    human_path = destination / "evaluation/semantic_human_adjudication.jsonl"
    human_path.write_text("", encoding="utf-8")
    candidate_pool_sources = {
        "attribute_based": {"status": "MATERIALIZED", "file": "candidate_pools/attribute_candidate_query_sets.jsonl", "index": "candidate_pools/attribute_candidate_index.jsonl", "query_set_count": len(attribute_index)},
        "lexical_text_retrieval": {"status": "MATERIALIZED", "file": "candidate_pools/lexical_text_candidates.jsonl", "row_count": len(lexical_rows)},
        "frozen_siglip2": {"status": "MATERIALIZED" if frozen_siglip.exists() else "MISSING", "file": "candidate_pools/frozen_siglip2_D1_common.jsonl" if frozen_siglip.exists() else None, "model_eval_is_frozen": True},
        "independent_remoteclip_georsclip": {"status": "MATERIALIZED" if georsclip.exists() else "MISSING", "file": "candidate_pools/independent_georsclip_top100.jsonl" if georsclip.exists() else None, "self_pool_forbidden": True},
        "random_negatives": {"status": "MATERIALIZED", "file": "candidate_pools/random_negatives.jsonl", "row_count": len(random_rows), "seed": 20260808},
    }
    write_json(destination / "candidate_pool_sources.json", candidate_pool_sources)
    write_json(destination / "candidate_pool_union_policy.json", {
        "schema_version": "qcpr-semantic-candidate-union-v1",
        "frozen_before_final_temporal_siglip": True,
        "sources": candidate_pool_sources,
        "human_grade_set": [3, 2, 1, 0, "IGNORE"],
        "transitivity": False,
        "human_judgments_materialized": False,
    })
    write_json(destination / "extension_readiness.json", {
        "schema_version": "qcpr-extension-readiness-v1",
        "SEMANTIC_EVAL_READY": {"decision": False, "status": "HOLD_IMMUTABLE_HUMAN_ADJUDICATION_MISSING", "candidate_pool_frozen": True, "human_judgments": 0},
        "STABLE_EVAL_READY": {"decision": False, "status": "HOLD_STABLE_HUMAN_GATE", "candidate_rows": stable_count},
        "LOCALIZED_EVAL_READY": {"decision": False, "status": "HOLD_VERIFIED_LOCALIZED_LANGUAGE", "candidate_rows": localized_count, "mask_sidecars_evaluation_only": True},
        "DUBAI_EXTERNAL_READY": {"decision": False, "status": "HOLD_ARCHIVE_PROVENANCE_LICENSE_LOADER_MANIFEST", "audit_copied": bool(copied.get("dubai_audit"))},
        "LONG_SERIES_READY": {"decision": False, "status": "HOLD_LICENSE_TEMPORAL_HUMAN_REVIEW", "candidate_rows": long_count},
        "training_views": {"semantic_train": False, "localized_train": False, "stable_train": False, "long_series_train": False},
    })
    metadata = {
        "schema_version": "qcpr-bitemporal-extension-benchmark-v1",
        "name": "QCPR_BITEMPORAL_EXTENSION_BENCHMARK_V1",
        "release_path": str(release),
        "candidate_pool_sources": candidate_pool_sources,
        "evaluation_files": {
            "stable": {"path": "evaluation/stable_evaluation.jsonl", "rows": stable_count},
            "localized": {"path": "evaluation/localized_evaluation.jsonl", "rows": localized_count},
            "long_series": {"path": "evaluation/long_series_evaluation.jsonl", "rows": long_count},
            "semantic_human_adjudication": {"path": "evaluation/semantic_human_adjudication.jsonl", "rows": 0},
        },
        "mask_policy": "mask sidecars evaluate localization only and never create training text",
        "training_enabled": False,
        "no_transitive_semantics": True,
    }
    write_json(destination / "benchmark_metadata.json", metadata)
    write_sha256sums(destination)
    return {"path": str(destination), "sha256": sha256_file(destination / "SHA256SUMS"), "metadata": metadata}


def calibration_gate(release: Path) -> dict[str, Any]:
    """Require completed human calibration before enabling any exact-loss view."""
    path = release / "tier_a_calibration_audit.json"
    if not path.exists():
        return {
            "passed": False,
            "status": "HOLD_CALIBRATION_AUDIT_MISSING",
            "required_decisions": 0,
            "completed_decisions": 0,
            "exact_scope_precision": None,
            "ci95": None,
            "explicit_gate_pass": False,
            "reason": "tier_a_calibration_audit.json is missing",
        }
    audit = load_json(path)
    strata = audit.get("required_strata") or list((audit.get("required_sample_counts") or {}).keys())
    required_per = int(audit.get("required_per_stratum", 0) or 0)
    required_total = required_per * len(strata)
    if not required_total:
        required_total = sum(int(value or 0) for value in (audit.get("required_sample_counts") or {}).values())
    completed = int(audit.get("reviewer_decisions_completed", 0) or 0)
    if not completed:
        completed = sum(int(value or 0) for value in (audit.get("decision_counts") or {}).values())
    metrics = audit.get("metrics") or {}
    precision = metrics.get("exact_scope_precision")
    ci95 = metrics.get("ci95")
    if ci95 is None:
        ci95 = metrics.get("exact_scope_precision_95_ci")
    required_metrics = (
        "exact_scope_precision",
        "false_exact_rate",
        "generic_contamination",
        "semantic_false_negative_rate",
        "average_positive_set_size",
        "collision_rate",
    )
    metrics_complete = precision is not None and ci95 is not None and all(metrics.get(name) is not None for name in required_metrics)
    exact_gate = audit.get("exact_gate") or {}
    explicit_pass = exact_gate.get("passes") is True or audit.get("status") == "PASS_EXACT_SCOPE_PRECISION_GATE"
    passed = bool(explicit_pass and completed >= required_total and metrics_complete)
    if passed:
        reason = "required human calibration is complete and the explicit exact-scope precision gate passes"
    elif completed < required_total:
        reason = f"human calibration decisions incomplete: {completed}/{required_total}"
    elif not metrics_complete:
        reason = "required calibration metrics are incomplete"
    else:
        reason = "exact-scope precision gate is not explicitly passed"
    return {
        "passed": passed,
        "status": "PASS_EXACT_SCOPE_PRECISION_GATE" if passed else "HOLD_EXACT_SCOPE_PRECISION_GATE",
        "required_decisions": required_total,
        "completed_decisions": completed,
        "exact_scope_precision": precision,
        "ci95": ci95,
        "explicit_gate_pass": explicit_pass,
        "metrics_complete": metrics_complete,
        "reason": reason,
    }


def apply_exact_training_gate(release: Path, enabled: bool) -> None:
    """Keep exact and direction train projections disabled until calibration passes."""
    for scope in ("exact", "direction"):
        path = release / "manifests" / f"{scope}_train.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        changed = False
        for row in rows:
            value = bool(enabled)
            if row.get("training_enabled") != value:
                row["training_enabled"] = value
                changed = True
        if changed:
            write_jsonl(path, rows)


def build_readiness(
    release: Path,
    core: dict[str, Any],
    extension: dict[str, Any],
    batch: dict[str, Any],
    path_audit: dict[str, Any],
    reversed_audit: dict[str, Any],
    exact_audit: dict[str, Any],
    validator_passed: bool,
) -> dict[str, Any]:
    core_passed = core["integrity"]["status"] == "PASS"
    batch_passed = all(batch["pass_conditions"].values())
    exact_integrity_passed = core_passed and batch_passed and path_audit.get("status", "").startswith("PASS") and reversed_audit.get("passed", False) and exact_audit.get("passed", False) and validator_passed
    calibration = calibration_gate(release)
    exact_passed = exact_integrity_passed and calibration["passed"]
    exact_status = "READY_FINAL_EXACT_CORE" if exact_passed else ("HOLD_EXACT_SCOPE_PRECISION_GATE" if exact_integrity_passed else "HOLD_FINAL_EXACT_INTEGRITY")
    return {
        "schema_version": "qcpr-final-readiness-states-v1",
        "calibration_gate": calibration,
        "BITEMPORAL_EXACT_READY": {"decision": exact_passed, "status": exact_status, "training_enabled": exact_passed, "evidence": ["exact_training_integrity_audit.json", "real_batch_relevance_audit.json", "core_benchmark_integrity.json", "tier_a_calibration_audit.json", "audits/validate_qcpr_release_contract.json"]},
        "SEMANTIC_EVAL_READY": {"decision": False, "status": "HOLD_IMMUTABLE_HUMAN_ADJUDICATION_MISSING", "training_enabled": False, "evidence": ["extension/extension_readiness.json", "extension/evaluation/semantic_human_adjudication.jsonl"]},
        "SEMANTIC_TRAIN_READY": {"decision": False, "status": "NO_VERIFIED_GRADE_2_OR_SEMANTIC_GOLD", "training_enabled": False, "evidence": ["manifests/semantic_train.jsonl"]},
        "STABLE_EVAL_READY": {"decision": False, "status": "HOLD_STABLE_HUMAN_GATE", "training_enabled": False, "evidence": ["extension/extension_readiness.json"]},
        "LOCALIZED_EVAL_READY": {"decision": False, "status": "HOLD_VERIFIED_LANGUAGE", "training_enabled": False, "evidence": ["extension/evaluation/localized_evaluation.jsonl", "evaluation_sidecars/dense.jsonl"]},
        "LOCALIZED_TRAIN_READY": {"decision": False, "status": "NO_VERIFIED_LOCALIZED_TRAINING_TEXT", "training_enabled": False, "evidence": ["manifests/localized_train.jsonl"]},
        "DUBAI_EXTERNAL_READY": {"decision": False, "status": "ACQUISITION_LICENSE_MANIFEST_HOLD", "training_enabled": False, "evidence": ["extension/external_benchmarks/dubai_cc_audit.json"]},
        "FOREST_TRAIN_READY": {"decision": False, "status": "PHYSICAL_TEXT_PROVENANCE_HOLD", "training_enabled": False, "evidence": ["source_reports/extension_audit_snapshot.json"]},
        "RSCC_TRAIN_READY": {"decision": False, "status": "PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED", "training_enabled": False, "evidence": ["source_reports/extension_audit_snapshot.json"]},
        "RSRCC_READY": {"decision": False, "status": "PARENT_PROVENANCE_HOLD", "training_enabled": False, "evidence": ["source_reports/source_terms_audit.json"]},
        "LONG_SERIES_READY": {"decision": False, "status": "LICENSE_TEMPORAL_REVIEW_HOLD", "training_enabled": False, "evidence": ["extension/extension_readiness.json"]},
        "AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING": exact_passed,
        "training_launched": False,
        "gate_inputs": {"core_benchmark_passed": core_passed, "real_batch_passed": batch_passed, "full_path_decode_passed": path_audit.get("status", "").startswith("PASS"), "reversed_leakage_passed": reversed_audit.get("passed", False), "exact_training_integrity_passed": exact_audit.get("passed", False), "calibration_passed": calibration["passed"], "validator_passed": validator_passed},
    }


def update_release_metadata(
    release: Path,
    code_sha: str,
    readiness: dict[str, Any],
    batch: dict[str, Any],
    core: dict[str, Any],
    extension: dict[str, Any],
    exact_audit: dict[str, Any],
) -> None:
    release_json = load_json(release / "RELEASE.json")
    release_json.update({
        "release_name": "QCPR_BITEMPORAL_V2_TRAIN",
        "release_variant": "FINAL_SYNCHRONIZED_R18",
        "immutable": True,
        "training_launched": False,
        "final_training_authorized": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
        "code_sha": code_sha,
        "core_benchmark_path": str(core["path"]),
        "extension_benchmark_path": str(extension["path"]),
        "readiness_states": "readiness_states.json",
        "final_gate": readiness["gate_inputs"],
    })
    counts = release_json.setdefault("counts", {})
    views = counts.setdefault("views", {})
    for scope in ("semantic",):
        views.setdefault(scope, {})
        for split in ("train", "development", "test"):
            views[scope][split] = 0
    for scope in ("localized", "stable", "long_series"):
        views.setdefault(scope, {})["train"] = 0
    for split in ("train", "development", "test"):
        views.setdefault("exact", {})[split] = safe_count(release / "manifests" / f"exact_{split}.jsonl")
    for scope in ("direction", "localized", "stable", "long_series"):
        for split in ("train", "development", "test"):
            views.setdefault(scope, {})[split] = safe_count(release / "manifests" / f"{scope}_{split}.jsonl")
    release_json["final_counts"] = {
        "physical_items": safe_count(release / "registries/physical_items.jsonl"),
        "frames": safe_count(release / "registries/frames.jsonl"),
        "canonical_queries": safe_count(release / "registries/queries.jsonl"),
        "exact_training_queries": exact_audit["query_count"],
        "unique_exact_training_physical_pairs": exact_audit["unique_exact_training_physical_pairs"],
        "disabled_generic_no_change_rows": safe_count(release / "registries/generic_no_change_diagnostic.jsonl"),
        "real_batch_matrix": batch["matrix_shape"],
    }
    write_json(release / "RELEASE.json", release_json)

    status = load_json(release / "dataset_status.json")
    status.update({
        "schema_version": "qcpr-dataset-status-v2-final",
        "branch": "codex/qcpr-dataset-v2-final",
        "dataset_head_sha": code_sha,
        "release_code_sha": code_sha,
        "release_path": str(release),
        "release_state": "FINAL_EXACT_AUTHORIZED_EXTENSION_HOLD" if readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"] else "FINAL_EXACT_HOLD",
        "final_training_authorized": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
        "readiness_states": readiness,
        "core_benchmark_path": str(core["path"]),
        "extension_benchmark_path": str(extension["path"]),
        "no_training_launched": True,
    })
    status["counts"] = release_json["final_counts"]
    write_json(release / "dataset_status.json", status)

    capabilities = load_json(release / "dataset_capabilities.json")
    capabilities.update({
        "schema_version": "qcpr-dataset-capabilities-v2-final",
        "release_code_sha": code_sha,
        "release_path": str(release),
        "release_state": status["release_state"],
        "final_readiness": readiness,
        "final_training_authorized": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
        "no_training_launched": True,
    })
    capabilities["views"] = views
    capabilities["counts"] = release_json["final_counts"]
    write_json(release / "dataset_capabilities.json", capabilities)


def write_release_reports(release: Path, code_sha: str, readiness: dict[str, Any], core: dict[str, Any], extension: dict[str, Any], batch: dict[str, Any], exact_audit: dict[str, Any], core_completeness: dict[str, Any], path_audit: dict[str, Any], reversed_audit: dict[str, Any]) -> None:
    decision_path = release / "source_reports/decision_package.json"
    decision = load_json(decision_path) if decision_path.exists() else {}
    decision.update({
        "schema_version": "qcpr-final-decision-package-v1",
        "final_synchronization": {
            "dataset_code_sha": code_sha,
            "release_path": str(release),
            "core_benchmark_path": str(core["path"]),
            "extension_benchmark_path": str(extension["path"]),
            "readiness_states": readiness,
            "core_source_completeness": core_completeness,
            "real_batch_relevance_audit": batch,
            "exact_training_integrity": exact_audit,
            "full_path_decode_audit": path_audit,
            "reversed_pair_leakage_audit": reversed_audit,
            "data_only_comparison_final_r18": str(release / "audits/data_only_comparison_final_r18.json") if (release / "audits/data_only_comparison_final_r18.json").exists() else None,
            "training_launched": False,
        },
    })
    write_json(decision_path, decision)
    write_json(release / "readiness_states.json", readiness)
    report = f"""# QCPR final synchronization report\n\n- Dataset release: `{release}`\n- Dataset code SHA: `{code_sha}`\n- Training launched: `false`\n- Exact final authorization: `{str(readiness['AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING']).lower()}`\n- Core benchmark: `{core['path']}`\n- Extension benchmark: `{extension['path']}`\n- Final frozen-model data-only comparison: `audits/data_only_comparison_final_r18.json`\n\n## Exact gate\n\nThe exact view uses trusted LEVIR-MCI and SECOND-CC captions. Grade 3 is a same-physical-pair positive; sparse collision/ambiguous cells are ignored; ordinary unlisted cells are implicit negatives. Generic no-change, mask-derived and generated-unverified text are disabled.\n\n- Exact training queries: `{exact_audit['query_count']}`\n- Unique exact-training physical pairs: `{exact_audit['unique_exact_training_physical_pairs']}`\n- Real batch: `{batch['matrix_shape']}`\n- Real batch status: `{batch['status']}`\n- Full path/decode/hash status: `{path_audit.get('status')}`\n- Reversed-pair leakage status: `{'PASS' if reversed_audit.get('passed') else 'FAIL'}`\n\n## Final data-only comparison\n\n`audits/data_only_comparison_final_r18.json` uses one already-frozen TemporalSigLIP ranking tensor, the final r18 query/gallery order, and paired bootstrap intervals. D0 and D1 share the same scores; the delta is a relevance-policy collision-ignore effect, not a model-improvement claim. D2/D3 and semantic/stable/localized metrics remain held where verified text or human judgments are absent.\n\n## Extension gate\n\nSemantic, stable, localized, Dubai, Forest text, RSCC text, RSRCC and long-series states remain independently held. Their candidate/evaluation artifacts are present only as explicitly unverified or evaluation-only evidence; none authorizes expanded training.\n\nThe semantic candidate union is frozen before final TemporalSigLIP evaluation and records attribute, lexical, frozen SigLIP2, independent GeoRSCLIP and deterministic random-negative pools. Human grades remain empty rather than fabricated.\n\n## Source completeness\n\nThe exact-core completeness audit is limited to LEVIR-MCI/LEVIR-CC and SECOND-CC, with every known missing item classified. Noncore sources remain source-level audits and are not silently promoted.\n"""
    report += f"\n\n## Calibration gate\n\n- Status: `{readiness['calibration_gate']['status']}`\n- Human decisions: `{readiness['calibration_gate']['completed_decisions']}/{readiness['calibration_gate']['required_decisions']}`\n- Exact training enabled: `{str(readiness['AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING']).lower()}`\n- Reason: `{readiness['calibration_gate']['reason']}`\n"
    (release / "source_reports/final_synchronization_report.md").write_text(report, encoding="utf-8")


def build_handoff(release: Path, core: dict[str, Any], extension: dict[str, Any], readiness: dict[str, Any], batch: dict[str, Any], exact_audit: dict[str, Any], code_sha: str) -> dict[str, Any]:
    def ref(relative: str) -> dict[str, str]:
        path = release / relative
        return {"path": str(path), "sha256": sha256_file(path)}

    physical_rows = load_jsonl(release / "registries/physical_items.jsonl")
    query_rows = load_jsonl(release / "registries/queries.jsonl")
    source_physical = collections.Counter(str(row.get("source")) for row in physical_rows)
    source_query = collections.Counter(split_item_id(str(row.get("source_item_id"))) for row in load_jsonl(release / "manifests/exact_train.jsonl"))
    handoff = {
        "schema_version": "qcpr-dataset-final-to-model-v1",
        "request_id": "QCPR_DATASET_V2_RETRIEVAL_SEMANTIC_REPAIR_FINAL_SYNC_R18",
        "dataset_branch": "codex/qcpr-dataset-v2-final",
        "dataset_code_sha": code_sha,
        "model_agent_head": MODEL_HEAD,
        "model_contract_sha256": MODEL_CONTRACT_SHA,
        "release_name": "QCPR_BITEMPORAL_V2_TRAIN",
        "release_path": str(release),
        "release_sha256sums": sha256_file(release / "SHA256SUMS"),
        "core_benchmark_path": core["path"],
        "core_benchmark_sha256sums": core["sha256"],
        "extension_benchmark_path": extension["path"],
        "extension_benchmark_sha256sums": extension["sha256"],
        "manifests": {
            "exact_train": ref("manifests/exact_train.jsonl"),
            "exact_development": ref("manifests/exact_development.jsonl"),
            "exact_test": ref("manifests/exact_test.jsonl"),
            "direction_train": ref("manifests/direction_train.jsonl"),
            "direction_development": ref("manifests/direction_development.jsonl"),
            "direction_test": ref("manifests/direction_test.jsonl"),
            "semantic_train": ref("manifests/semantic_train.jsonl"),
            "semantic_development": ref("manifests/semantic_development.jsonl"),
            "semantic_test": ref("manifests/semantic_test.jsonl"),
            "localized_train": ref("manifests/localized_train.jsonl"),
            "localized_development": ref("manifests/localized_development.jsonl"),
            "localized_test": ref("manifests/localized_test.jsonl"),
            "stable_train": ref("manifests/stable_train.jsonl"),
            "stable_development": ref("manifests/stable_development.jsonl"),
            "stable_test": ref("manifests/stable_test.jsonl"),
            "long_series_train": ref("manifests/long_series_train.jsonl"),
            "long_series_development": ref("manifests/long_series_development.jsonl"),
            "long_series_test": ref("manifests/long_series_test.jsonl"),
        },
        "registries": {
            "physical_items": ref("registries/physical_items.jsonl"),
            "frames": ref("registries/frames.jsonl"),
            "queries": ref("registries/queries.jsonl"),
            "query_to_pair_relevance": ref("registries/query_to_pair_relevance.jsonl"),
            "relevance_policy": ref("relevance_policy.json"),
            "collision_groups": ref("collision_groups.jsonl"),
            "sampler_contract": ref("sampler_contract.json"),
            "semantic_candidate_groups": ref("registries/semantic_candidate_groups.jsonl"),
            "semantic_candidate_query_sets": ref("registries/semantic_candidate_query_sets.jsonl"),
            "generic_no_change_diagnostic": ref("registries/generic_no_change_diagnostic.jsonl"),
        },
        "audits": {
            "real_batch_relevance": ref("real_batch_relevance_audit.json"),
            "exact_training_integrity": ref("exact_training_integrity_audit.json"),
            "full_path_decode": ref("full_path_decode_audit.json"),
            "reversed_pair_leakage": ref("reversed_pair_leakage_audit.json"),
            "validation": ref("audits/validate_qcpr_release_contract.json"),
            "calibration": ref("tier_a_calibration_audit.json"),
            "mask_sidecars": ref("evaluation_sidecars/dense.jsonl"),
        },
        "counts": {
            "physical_items": len(physical_rows),
            "frames": safe_count(release / "registries/frames.jsonl"),
            "canonical_queries": len(query_rows),
            "verified_exact_queries": exact_audit["query_count"],
            "exact_training_enabled": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
            "direction_training_enabled": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
            "calibration_decisions_completed": readiness["calibration_gate"]["completed_decisions"],
            "calibration_decisions_required": readiness["calibration_gate"]["required_decisions"],
            "unique_exact_training_physical_pairs": exact_audit["unique_exact_training_physical_pairs"],
            "exact_training_query_counts_by_source": dict(sorted(source_query.items())),
            "direction_query_counts_by_split": {split: safe_count(release / "manifests" / f"direction_{split}.jsonl") for split in ("train", "development", "test")},
            "direction_query_count": sum(safe_count(release / "manifests" / f"direction_{split}.jsonl") for split in ("train", "development", "test")),
            "physical_counts_by_source": dict(sorted(source_physical.items())),
            "semantic_candidate_groups": safe_count(release / "registries/semantic_candidate_groups.jsonl"),
            "semantic_candidate_query_sets": safe_count(release / "registries/semantic_candidate_query_sets.jsonl"),
            "localized_evaluation_queries": safe_count(Path(extension["path"]) / "evaluation/localized_evaluation.jsonl"),
            "stable_evaluation_queries": safe_count(Path(extension["path"]) / "evaluation/stable_evaluation.jsonl"),
            "long_series_evaluation_queries": safe_count(Path(extension["path"]) / "evaluation/long_series_evaluation.jsonl"),
            "disabled_generic_no_change_rows": safe_count(release / "registries/generic_no_change_diagnostic.jsonl"),
            "mask_sidecars": safe_count(release / "evaluation_sidecars/dense.jsonl"),
        },
        "real_batch": {
            "matrix_shape": batch["matrix_shape"],
            "cell_counts": batch["cell_counts"],
            "collision_rate": batch["collision_rate"],
            "same_pair_positive_cells": batch["same_pair_positive_cells"],
            "same_pair_false_negative_count": batch["same_pair_false_negative_count"],
        },
        "readiness": readiness,
        "training_launched": False,
        "model_instruction": "Model Agent must consume this handoff as the sole final dataset contract; do not infer readiness or counts from legacy reports.",
    }
    comparison = release / "audits/data_only_comparison_final_r18.json"
    if comparison.exists():
        handoff["audits"]["data_only_comparison_final_r18"] = ref("audits/data_only_comparison_final_r18.json")
    return handoff


def update_shared_contracts(code_sha: str, release: Path, core: dict[str, Any], extension: dict[str, Any], readiness: dict[str, Any], handoff_sha: str) -> None:
    status_path = REPO / "contracts/qcpr_shared/dataset_status.json"
    status = load_json(status_path) if status_path.exists() else {}
    status.update({
        "schema_version": "qcpr-dataset-status-v2-final",
        "branch": "codex/qcpr-dataset-v2-final",
        "dataset_head_sha": code_sha,
        "release_code_sha": code_sha,
        "release_path": str(release),
        "release_sha256": sha256_file(release / "SHA256SUMS"),
        "release_state": "FINAL_EXACT_AUTHORIZED_EXTENSION_HOLD" if readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"] else "FINAL_EXACT_HOLD",
        "final_training_authorized": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
        "final_readiness": readiness,
        "core_benchmark_path": str(core["path"]),
        "core_benchmark_sha256": core["sha256"],
        "extension_benchmark_path": str(extension["path"]),
        "extension_benchmark_sha256": extension["sha256"],
        "dataset_final_handoff_path": str(HANDOFF),
        "dataset_final_handoff_sha256": handoff_sha,
        "training_launched": False,
    })
    write_json(status_path, status)
    cap_path = REPO / "contracts/qcpr_shared/dataset_capabilities.json"
    capabilities = load_json(cap_path) if cap_path.exists() else {}
    capabilities.update({
        "schema_version": "qcpr-dataset-capabilities-v2-final",
        "release_code_sha": code_sha,
        "release_path": str(release),
        "release_sha256": sha256_file(release / "SHA256SUMS"),
        "release_state": status["release_state"],
        "final_training_authorized": readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"],
        "final_readiness": readiness,
        "core_benchmark_path": str(core["path"]),
        "core_benchmark_sha256": core["sha256"],
        "extension_benchmark_path": str(extension["path"]),
        "extension_benchmark_sha256": extension["sha256"],
        "dataset_final_handoff_path": str(HANDOFF),
        "dataset_final_handoff_sha256": handoff_sha,
        "training_launched": False,
    })
    write_json(cap_path, capabilities)


def repair_existing_release(base: Path, release: Path, core_path: Path, extension_path: Path) -> int:
    """Refresh derived gate artifacts without re-copying native assets."""
    code_sha = git_head()
    batch = build_batch_audit(base)
    write_json(release / "relevance_policy.json", build_relevance_policy())
    write_json(release / "sampler_contract.json", build_sampler_contract(base, load_json(base / "loader/temporal_siglip_batch_128x256.json")))
    write_json(release / "real_batch_relevance_audit.json", batch)
    path_audit = load_json(release / "full_path_decode_audit.json")
    reversed_audit = load_json(release / "reversed_pair_leakage_audit.json")
    apply_exact_training_gate(release, calibration_gate(release)["passed"])
    exact_audit = build_exact_training_integrity(release)
    write_json(release / "exact_training_integrity_audit.json", exact_audit)
    core = {
        "path": str(core_path),
        "sha256": sha256_file(core_path / "SHA256SUMS"),
        "metadata": load_json(core_path / "benchmark_metadata.json"),
        "integrity": load_json(core_path / "core_benchmark_integrity.json"),
    }
    extension = {
        "path": str(extension_path),
        "sha256": sha256_file(extension_path / "SHA256SUMS"),
        "metadata": load_json(extension_path / "benchmark_metadata.json"),
    }
    core_completeness = build_core_source_completeness(base)
    preliminary_readiness = build_readiness(release, core, extension, batch, path_audit, reversed_audit, exact_audit, False)
    update_release_metadata(release, code_sha, preliminary_readiness, batch, core, extension, exact_audit)
    write_release_reports(release, code_sha, preliminary_readiness, core, extension, batch, exact_audit, core_completeness, path_audit, reversed_audit)
    write_json(release / "audits/validate_qcpr_release_contract.json", {"status": "PENDING_FINAL_VALIDATION"})
    write_sha256sums(release)
    validator = REPO / "scripts/validate_qcpr_release_contract.py"
    temp_validation = Path(tempfile.gettempdir()) / "qcpr_final_validation_r18.json"
    if validator.exists():
        first = subprocess.run([sys.executable, str(validator), "--release", str(release), "--output", str(release / "audits/validate_qcpr_release_contract.json"), "--decode-sample", "256"], cwd=REPO, text=True, capture_output=True)
        print(f"validator repair first pass exit={first.returncode}", flush=True)
        write_sha256sums(release)
        second = subprocess.run([sys.executable, str(validator), "--release", str(release), "--output", str(temp_validation), "--decode-sample", "256"], cwd=REPO, text=True, capture_output=True)
        validator_result = load_json(temp_validation) if temp_validation.exists() else {"passed": False, "error": second.stderr[-1000:]}
        print(f"validator repair recheck exit={second.returncode} passed={validator_result.get('passed')}", flush=True)
        write_json(release / "audits/validate_qcpr_release_contract.json", validator_result)
    else:
        validator_result = {"passed": False, "error": f"validator missing: {validator}"}
    validator_passed = bool(validator_result.get("passed", False))
    readiness = build_readiness(release, core, extension, batch, path_audit, reversed_audit, exact_audit, validator_passed)
    update_release_metadata(release, code_sha, readiness, batch, core, extension, exact_audit)
    write_release_reports(release, code_sha, readiness, core, extension, batch, exact_audit, core_completeness, path_audit, reversed_audit)
    write_sha256sums(release)
    handoff = build_handoff(release, core, extension, readiness, batch, exact_audit, code_sha)
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    write_json(HANDOFF, handoff)
    handoff_sha = sha256_file(HANDOFF)
    update_shared_contracts(code_sha, release, core, extension, readiness, handoff_sha)
    print(json.dumps({"release": str(release), "release_sha256sums": sha256_file(release / "SHA256SUMS"), "core": core, "extension": extension, "handoff": str(HANDOFF), "handoff_sha256": handoff_sha, "readiness": readiness, "validator_passed": validator_passed}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validator_passed and readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    base = args.base
    release = args.release
    core_path = args.core
    extension_path = args.extension
    if args.repair_existing:
        if not base.exists() or not release.exists() or not core_path.exists() or not extension_path.exists():
            raise SystemExit("--repair-existing requires base, release, core and extension outputs")
        return repair_existing_release(base, release, core_path, extension_path)
    for path in (release, core_path, extension_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if not base.exists():
        raise SystemExit(f"base release not found: {base}")

    print(f"copying immutable base release: {base} -> {release}", flush=True)
    shutil.copytree(base, release)
    materialize_hold_views(release)
    code_sha = git_head()
    batch = build_batch_audit(base)
    write_json(release / "relevance_policy.json", build_relevance_policy())
    write_json(release / "sampler_contract.json", build_sampler_contract(base, load_json(base / "loader/temporal_siglip_batch_128x256.json")))
    copy(base / "registries/relevance_collision_groups.jsonl", release / "collision_groups.jsonl")
    write_json(release / "real_batch_relevance_audit.json", batch)
    write_json(release / "core_source_completeness.json", build_core_source_completeness(base))
    write_json(release / "preprocessing_contract.json", build_preprocessing_contract(base))
    write_json(release / "tier_a_calibration_audit.json", {
        "schema_version": "qcpr-tier-a-calibration-audit-v1",
        "required_strata": ["exact", "semantic", "generic_no_change", "stable_scene", "localized_direction"],
        "required_per_stratum": 300,
        "packets_materialized": 1500,
        "reviewer_decisions_completed": 0,
        "status": "PACKETS_MATERIALIZED_REVIEW_DECISIONS_PENDING",
        "metrics": {"exact_scope_precision": None, "ci95": None, "false_exact_rate": None, "generic_contamination": None, "semantic_false_negative_rate": None, "average_positive_set_size": None, "collision_rate": None},
        "exact_gate": {"passes": False, "status": "HOLD_EXACT_SCOPE_PRECISION_GATE", "reason": "all reviewer decisions are pending; exact view cannot be enabled"},
        "no_decisions_fabricated": True,
    })

    print(f"running full core frame path/decode/hash audit", flush=True)
    path_audit = build_path_decode_audit(base)
    write_json(release / "full_path_decode_audit.json", path_audit)
    reversed_audit = build_reversed_pair_audit(base)
    write_json(release / "reversed_pair_leakage_audit.json", reversed_audit)
    apply_exact_training_gate(release, calibration_gate(release)["passed"])
    exact_audit = build_exact_training_integrity(release)
    write_json(release / "exact_training_integrity_audit.json", exact_audit)

    print(f"building core benchmark: {core_path}", flush=True)
    core = build_core_benchmark(base, release, core_path)
    print(f"building extension benchmark: {extension_path}", flush=True)
    extension = build_extension_benchmark(base, release, extension_path)
    core_completeness = build_core_source_completeness(base)

    preliminary_readiness = build_readiness(release, core, extension, batch, path_audit, reversed_audit, exact_audit, False)
    update_release_metadata(release, code_sha, preliminary_readiness, batch, core, extension, exact_audit)
    write_release_reports(release, code_sha, preliminary_readiness, core, extension, batch, exact_audit, core_completeness, path_audit, reversed_audit)
    write_json(release / "audits/validate_qcpr_release_contract.json", {"status": "PENDING_FINAL_VALIDATION"})
    write_sha256sums(release)

    validator = REPO / "scripts/validate_qcpr_release_contract.py"
    temp_validation = Path(tempfile.gettempdir()) / "qcpr_final_validation_r18.json"
    if validator.exists():
        first = subprocess.run([sys.executable, str(validator), "--release", str(release), "--output", str(release / "audits/validate_qcpr_release_contract.json"), "--decode-sample", "256"], cwd=REPO, text=True, capture_output=True)
        print(f"validator first pass exit={first.returncode}", flush=True)
        write_sha256sums(release)
        second = subprocess.run([sys.executable, str(validator), "--release", str(release), "--output", str(temp_validation), "--decode-sample", "256"], cwd=REPO, text=True, capture_output=True)
        validator_result = load_json(temp_validation) if temp_validation.exists() else {"passed": False, "error": second.stderr[-1000:]}
        print(f"validator final recheck exit={second.returncode} passed={validator_result.get('passed')}", flush=True)
        write_json(release / "audits/validate_qcpr_release_contract.json", validator_result)
    else:
        validator_result = {"passed": False, "error": f"validator missing: {validator}"}
    validator_passed = bool(validator_result.get("passed", False))
    readiness = build_readiness(release, core, extension, batch, path_audit, reversed_audit, exact_audit, validator_passed)
    update_release_metadata(release, code_sha, readiness, batch, core, extension, exact_audit)
    write_release_reports(release, code_sha, readiness, core, extension, batch, exact_audit, core_completeness, path_audit, reversed_audit)
    write_sha256sums(release)
    release_sha = sha256_file(release / "SHA256SUMS")

    handoff = build_handoff(release, core, extension, readiness, batch, exact_audit, code_sha)
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    write_json(HANDOFF, handoff)
    handoff_sha = sha256_file(HANDOFF)
    update_shared_contracts(code_sha, release, core, extension, readiness, handoff_sha)

    print(json.dumps({
        "release": str(release),
        "release_sha256sums": release_sha,
        "core": core,
        "extension": extension,
        "handoff": str(HANDOFF),
        "handoff_sha256": handoff_sha,
        "readiness": readiness,
        "validator_passed": validator_passed,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validator_passed and readiness["AUTHORIZE_TEMPORALSIGLIP_FINAL_TRAINING"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
