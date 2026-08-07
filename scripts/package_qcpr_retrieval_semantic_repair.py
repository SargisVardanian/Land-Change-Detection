#!/usr/bin/env python3
"""Package an immutable QCPR retrieval-semantic repair release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_MANIFESTS = [
    f"{scope}_{split}.jsonl"
    for scope in ("exact", "semantic", "localized", "long_series", "direction", "stable")
    for split in ("train", "development", "test")
]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rsrcc_parent_overlap_audit(
    parent_registry: Path,
    child_manifest: Path,
    supplemental_parent_registry: Path | None = None,
) -> dict[str, Any]:
    """Compare RSRCC image hashes with the physical parent registry.

    RSRCC is derived from LEVIR-CD according to the published paper, so a
    child-source license claim is not enough to promote its image text into
    training.  This audit is deliberately hash-based and does not infer
    semantic relevance from filenames or event identity.
    """
    parent_rows = read_jsonl(parent_registry)
    parent_paths = [str(parent_registry)]
    if supplemental_parent_registry is not None and supplemental_parent_registry.exists():
        supplemental_rows = read_jsonl(supplemental_parent_registry)
        parent_rows.extend(supplemental_rows)
        parent_paths.append(str(supplemental_parent_registry))
    parent_item_ids = {str(row.get("item_id")) for row in parent_rows if row.get("item_id")}
    parent_frame_hashes = {
        str(frame.get("sha256"))
        for row in parent_rows
        for frame in (row.get("frames") or [])
        if isinstance(frame, dict) and frame.get("sha256")
    }
    child_rows = read_jsonl(child_manifest)
    child_hashes = {
        str(row.get("sha256"))
        for row in child_rows
        if row.get("sha256")
    }
    shared = sorted(child_hashes & parent_frame_hashes)
    return {
        "schema_version": "qcpr-rsrcc-parent-overlap-audit-v1",
        "status": "PASS_NO_SHARED_FRAME_HASHES" if not shared else "HOLD_SHARED_PARENT_FRAME_HASHES",
        "child_dataset": "google/RSRCC",
        "child_manifest": "source_reports/retrieval_semantic_repair/rsrcc_physical_asset_manifest.jsonl",
        "child_manifest_row_count": len(child_rows),
        "child_unique_sha256_count": len(child_hashes),
        "parent_registry_paths": parent_paths,
        "parent_item_count": len(parent_item_ids),
        "parent_frame_hash_count": len(parent_frame_hashes),
        "shared_frame_hash_count": len(shared),
        "shared_frame_hash_examples": shared[:20],
    }


def write_relevance_graph(path: Path, query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Materialize one deterministic positive edge per query/item relation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    edge_count = 0
    with path.open("wb") as handle:
        ordered_queries = sorted(query_rows, key=lambda row: str(row.get("query_id")))
        for query in ordered_queries:
            query_id = str(query.get("query_id") or "")
            query_scope = str(query.get("query_scope") or "")
            provenance = query.get("provenance") if isinstance(query.get("provenance"), dict) else {}
            grades = query.get("graded_relevance") if isinstance(query.get("graded_relevance"), dict) else {}
            positive_ids = sorted({str(item_id) for item_id in query.get("positive_item_ids") or []})
            for item_id in positive_ids:
                edge = {
                    "schema_version": "qcpr-relevance-graph-edge-v1",
                    "query_id": query_id,
                    "item_id": item_id,
                    "source_item_id": query.get("source_item_id"),
                    "relevance_grade": int(grades[item_id]),
                    "query_scope": query_scope,
                    "purpose": query.get("purpose"),
                    "split": query.get("split"),
                    "verification": query.get("verification"),
                    "training_enabled": bool(query.get("training_enabled")),
                    "diagnostic_only": query_scope == "generic_no_change",
                    "semantic_group_id": provenance.get("semantic_group_id"),
                }
                payload = (json.dumps(edge, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                edge_count += 1
    return {
        "schema_version": "qcpr-relevance-graph-edge-v1",
        "path": str(path),
        "edge_count": edge_count,
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def merge_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(path, destination)


def write_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        checksums[str(path.relative_to(root))] = sha256(path)
    text = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items()))
    (root / "SHA256SUMS").write_text(text, encoding="utf-8")
    return checksums


def verify_checksums(root: Path) -> dict[str, Any]:
    failures = []
    rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for row in rows:
        if "  " not in row:
            failures.append({"row": row, "reason": "invalid_checksum_row"})
            continue
        expected, relative = row.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual, "reason": "mismatch"})
    return {"entries": len(rows), "failures": failures, "passed": not failures}


def write_decision_package(
    root: Path,
    *,
    branch: str,
    code_sha: str,
    release_path: Path,
    items: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    handoff: dict[str, Any],
    view_readiness: dict[str, Any],
    data_only_comparison: dict[str, Any],
    integrity: dict[str, Any],
) -> None:
    """Regenerate the legacy-compatible decision package from release rows.

    The retrieval repair package adds physical sources after copying the base
    release.  Reusing the base decision package would therefore leave stale
    item/frame/query counts next to the immutable registries.
    """
    source_registry = read_jsonl(root / "registries/source_registry.jsonl")
    by_source = Counter(str(row.get("source")) for row in items)
    by_scope = Counter(str(row.get("query_scope")) for row in queries)
    by_verification = Counter(str(row.get("verification")) for row in queries)
    summary = {
        "schema_version": "qcpr-dataset-v2-decision-package-v1",
        "branch": branch,
        "code_sha": code_sha,
        "release_path": str(release_path),
        "physical_item_counts": dict(sorted(by_source.items())),
        "physical_item_total": len(items),
        "sequence_count": sum(row.get("item_type") == "sequence" for row in items),
        "frame_count": sum(len(row.get("frames", [])) for row in items),
        "query_counts": dict(sorted(by_scope.items())),
        "verification_counts": dict(sorted(by_verification.items())),
        "integrity": dict(integrity),
        "source_status": source_registry,
        "readiness": {
            "release_state": "DATA_QUALITY_HOLD",
            "training_authorized": False,
            "main_training_allowed": False,
            "views": view_readiness.get("views", {}),
            "exact_gate": handoff.get("exact_gate"),
            "stable_gate": handoff.get("stable_gate"),
            "localized_gate": handoff.get("localized_gate"),
            "long_series_gate": handoff.get("long_series_gate"),
            "data_only_comparison_status": data_only_comparison.get("status"),
        },
    }
    write_json(root / "source_reports/decision_package.json", summary)
    lines = [
        "# Dataset-v2 retrieval-semantic repair decision package",
        "",
        f"- Branch: `{branch}`",
        f"- Code SHA: `{code_sha}`",
        f"- Release: `{release_path}`",
        f"- Physical items: `{summary['physical_item_total']}`",
        f"- Frames: `{summary['frame_count']}`",
        "",
        "## Query counts",
        "",
    ]
    for key, value in sorted(summary["query_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Readiness", ""])
    for key, value in sorted(summary["readiness"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Integrity", "", f"- passed: `{summary['integrity'].get('passed')}`"])
    (root / "source_reports/decision_package.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-branch", default="codex/qcpr-dataset-v2-final")
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--data-only-comparison", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing release: {args.output}")
    if not args.base_release.is_dir() or not args.repair_output.is_dir():
        raise SystemExit("base release and repair output must be directories")
    shutil.copytree(args.base_release, args.output)

    # Candidate manifests and registries replace only the retrieval views;
    # physical base assets remain byte-for-byte copied from the prior release.
    merge_tree(args.repair_output / "manifests", args.output / "manifests")
    copy_file(args.repair_output / "registries/repaired_queries.jsonl", args.output / "registries/queries.jsonl")
    copy_file(args.repair_output / "registries/query_purpose_registry.jsonl", args.output / "registries/query_purpose_registry.jsonl")
    copy_file(args.repair_output / "registries/semantic_groups.jsonl", args.output / "registries/semantic_groups.jsonl")
    copy_file(args.repair_output / "registries/generic_no_change_diagnostic.jsonl", args.output / "registries/generic_no_change_diagnostic.jsonl")
    merge_tree(args.repair_output / "evaluation_sidecars", args.output / "evaluation_sidecars")
    merge_tree(args.repair_output / "audits", args.output / "audits/retrieval_semantic_repair")
    merge_tree(args.repair_output / "source_reports", args.output / "source_reports/retrieval_semantic_repair")

    # Keep the legacy root RSRCC audit synchronized with the newly acquired
    # physical manifest.  The source remains license-held and text-disabled,
    # but stale "asset acquisition incomplete" evidence must not survive in
    # the immutable release next to the current acquisition audit.
    rsrcc_acquisition = read_json(args.repair_output / "source_reports/rsrcc_physical_asset_acquisition.json", {})
    rsrcc_validation = read_json(args.repair_output / "source_reports/rsrcc_physical_manifest_validation.json", {})
    rsrcc_manifest = args.repair_output / "source_reports/rsrcc_physical_asset_manifest.jsonl"
    rsrcc_gate_report = read_json(args.repair_output / "source_reports/official_source_acquisition_audit.json", {})
    rsrcc_gate = (rsrcc_gate_report.get("sources") or {}).get("RSRCC", {})
    if rsrcc_acquisition and rsrcc_validation and rsrcc_manifest.exists():
        parent_overlap = build_rsrcc_parent_overlap_audit(
            args.base_release / "registries/physical_items.jsonl",
            rsrcc_manifest,
            args.repair_output / "registries/forest_physical_items.jsonl",
        )
        parent_provenance = {
            "schema_version": "qcpr-rsrcc-parent-provenance-audit-v1",
            "status": "HOLD_PARENT_DATA_TERMS_REVIEW",
            "child_dataset": "google/RSRCC",
            "child_license_claim": "Apache-2.0 (official HF dataset card)",
            "child_license_source": "https://huggingface.co/datasets/google/RSRCC",
            "parent_dataset": "LEVIR-CD",
            "parent_provenance_source": "https://arxiv.org/abs/2604.20623",
            "parent_terms_source": "https://justchenhao.github.io/LEVIR/",
            "parent_terms_status": "Official parent page states Google Earth terms apply and academic-only/non-commercial use.",
            "overlap_audit": parent_overlap,
            "training_enabled": False,
        }
        rsrcc_path = args.output / "source_reports/rsrcc_source_audit.json"
        rsrcc_audit = read_json(rsrcc_path, {})
        rsrcc_physical = dict(rsrcc_audit.get("physical_asset_audit") or {})
        rsrcc_physical.update({
            "asset_manifest": "source_reports/retrieval_semantic_repair/rsrcc_physical_asset_manifest.jsonl",
            "available_asset_count": rsrcc_acquisition.get("available_asset_count"),
            "missing_asset_count": rsrcc_acquisition.get("missing_asset_count"),
            "downloaded_asset_count": rsrcc_acquisition.get("downloaded_asset_count"),
            "unique_asset_count": rsrcc_acquisition.get("unique_asset_count"),
            "revision": rsrcc_acquisition.get("revision"),
            "status": "PHYSICAL_ASSET_ACQUISITION_COMPLETE",
            "training_enabled": False,
            "hash_validation": {
                "passed": bool(rsrcc_validation.get("passed")),
                "asset_count": rsrcc_validation.get("asset_count"),
                "missing_count": rsrcc_validation.get("missing_count"),
                "hash_mismatch_count": rsrcc_validation.get("hash_mismatch_count"),
            },
            "parent_overlap_audit": parent_overlap["status"],
            "parent_overlap_details": parent_overlap,
        })
        rsrcc_audit.update({
            "status": "PHYSICAL_ASSETS_COMPLETE_LICENSE_HOLD",
            "download_observation": "Pinned official HF physical acquisition complete; generated text remains evaluation-only.",
            "repository_image_file_count": rsrcc_acquisition.get("unique_asset_count"),
            "revision": rsrcc_acquisition.get("revision"),
            "physical_asset_audit": rsrcc_physical,
            "physical_asset_manifest_sha256": sha256(rsrcc_manifest),
            "physical_asset_validation": {
                "path": "source_reports/retrieval_semantic_repair/rsrcc_physical_manifest_validation.json",
                "passed": bool(rsrcc_validation.get("passed")),
                "asset_count": rsrcc_validation.get("asset_count"),
                "missing_count": rsrcc_validation.get("missing_count"),
                "hash_mismatch_count": rsrcc_validation.get("hash_mismatch_count"),
            },
            "license_status": rsrcc_gate.get("license_status") or rsrcc_audit.get("license_status"),
            "training_enabled": False,
            "integration_status": "NOT_INTEGRATED_LICENSE_AND_PARENT_DATA_REVIEW_REQUIRED",
            "parent_overlap_audit": parent_overlap["status"],
            "parent_provenance_audit": parent_provenance,
        })
        write_json(rsrcc_path, rsrcc_audit)
        write_json(args.output / "source_reports/rsrcc_parent_provenance_audit.json", parent_provenance)
    # Keep the legacy root source audit aligned with the canonical repair audit.
    # The legacy schema remains for older consumers, while the nested snapshot
    # records the current five-gate status without silently promoting a source.
    legacy_official_path = args.output / "source_reports/official_source_audit.json"
    legacy_official = read_json(legacy_official_path, {})
    legacy_rows = legacy_official.get("sources")
    current_source_rows = rsrcc_gate_report.get("sources") or {}
    if isinstance(legacy_rows, list) and isinstance(current_source_rows, dict):
        source_aliases = {"SpaceNet 7": "SpaceNet-7"}
        for source_name, current in current_source_rows.items():
            legacy_name = source_aliases.get(source_name, source_name)
            for row in legacy_rows:
                if str(row.get("source_dataset")) != legacy_name:
                    continue
                row["state"] = current.get("status", row.get("state"))
                row["license_status"] = current.get("license_status", row.get("license_status"))
                row["training_enabled"] = bool(current.get("training_enabled"))
                row["retrieval_semantic_repair"] = {
                    "status": current.get("status"),
                    "integration_prerequisites": current.get("integration_prerequisites", {}),
                    "missing_prerequisites": current.get("missing_prerequisites", []),
                    "access_audit": current.get("access_audit", {}),
                }
                break
        legacy_official["retrieval_semantic_repair"] = {
            "status": rsrcc_gate_report.get("status"),
            "canonical_path": "source_reports/retrieval_semantic_repair/official_source_acquisition_audit.json",
            "ready_sources": rsrcc_gate_report.get("ready_sources", []),
            "source_statuses": {
                name: value.get("status")
                for name, value in sorted(current_source_rows.items())
            },
        }
        write_json(legacy_official_path, legacy_official)
    # Keep Forest's deterministic scene-component split as a canonical source
    # report as well as a repair-package artifact.  The physical split is
    # materialized, but captions remain held until provenance/review gates
    # pass; do not silently leave the legacy empty split files in the release.
    forest_repair_dir = args.repair_output / "source_reports/forest"
    forest_release_dir = args.output / "source_reports/forest"
    forest_split_audit = read_json(forest_repair_dir / "scene_disjoint_split_audit.json", {})
    forest_caption_audit = read_json(forest_repair_dir / "caption_level_audit.json", {})
    if forest_split_audit:
        for artifact in (
            "scene_disjoint_train.jsonl",
            "scene_disjoint_development.jsonl",
            "scene_disjoint_test.jsonl",
            "caption_level_audit.json",
            "scene_disjoint_split_audit.json",
        ):
            source = forest_repair_dir / artifact
            if source.exists():
                copy_file(source, forest_release_dir / artifact)
        physical_source = args.repair_output / "registries/forest_physical_items.jsonl"
        if physical_source.exists():
            copy_file(physical_source, forest_release_dir / "forest_physical_registry_scene_disjoint.jsonl")
        legacy_forest_audit = read_json(forest_release_dir / "forest_split_integrity.json", {})
        legacy_forest_audit.update({
            "status": "MATERIALIZED_SCENE_COMPONENT_DISJOINT_CANDIDATE",
            "promotion_status": forest_split_audit.get("promotion_status", "HOLD_CAPTION_PROVENANCE_AND_HUMAN_REVIEW"),
            "component_count": forest_split_audit.get("component_count"),
            "cross_split_component_overlap": forest_split_audit.get("cross_split_component_overlap"),
            "materialized_pair_counts": forest_split_audit.get("split_pair_counts", {}),
            "materialized_component_counts": forest_split_audit.get("split_component_counts", {}),
            "materialized_split_manifests": [
                "source_reports/forest/scene_disjoint_train.jsonl",
                "source_reports/forest/scene_disjoint_development.jsonl",
                "source_reports/forest/scene_disjoint_test.jsonl",
            ],
            "caption_level_audit": forest_caption_audit,
            "training_enabled": False,
        })
        write_json(forest_release_dir / "forest_split_integrity.json", legacy_forest_audit)
    # Canonicalize the TAMMs physical/annotation audit next to the physical
    # source reports.  The fields are schema-complete but intentionally
    # unverified; this makes coverage and the training hold auditable without
    # turning generated temporal descriptions into supervision.
    tamm_repair_audit = read_json(args.repair_output / "source_reports/tamms_long_series_audit.json", {})
    tamm_annotation_source = args.repair_output / "source_reports/tamms_long_series_annotation_audit.jsonl"
    if tamm_repair_audit and tamm_annotation_source.exists():
        tamm_release_audit_path = args.output / "source_reports/tamms_long_series_audit.json"
        tamm_release_audit = read_json(tamm_release_audit_path, {})
        tamm_annotation_rows = read_jsonl(tamm_annotation_source)
        tamm_physical_rows = [
            row
            for row in read_jsonl(args.base_release / "registries/physical_items.jsonl")
            if str(row.get("source", "")).casefold() == "tamms"
        ]
        annotation_fields = ("onset", "duration", "gradual_or_abrupt", "relevant_frame_range")
        field_coverage = {
            field: {
                "rows": len(tamm_annotation_rows),
                "non_null": sum(row.get(field) is not None for row in tamm_annotation_rows),
                "null_or_unverified": sum(row.get(field) is None for row in tamm_annotation_rows),
            }
            for field in annotation_fields
        }
        tamm_release_audit.update({
            "physical_inventory_count": len(tamm_physical_rows),
            "physical_split_counts": {
                split: sum(row.get("split") == split for row in tamm_physical_rows)
                for split in ("train", "development", "test")
            },
            "annotation_field_coverage": field_coverage,
            "annotation_rows_materialized": len(tamm_annotation_rows),
            "verified_long_series_query_count": 0,
            "training_enabled": False,
            "text_status": "GENERATED_UNVERIFIED_HUMAN_REVIEW_REQUIRED",
        })
        copy_file(tamm_annotation_source, args.output / "source_reports/tamms_long_series_annotation_audit.jsonl")
        write_json(tamm_release_audit_path, tamm_release_audit)
    for packet in sorted((args.repair_output / "source_reports/review_samples").glob("*.jsonl")):
        copy_file(packet, args.output / "source_reports/human_review_packets" / packet.name)
    merge_tree(args.repair_output / "handoff", args.output / "handoff/retrieval_semantic_repair")

    # Forest physical assets are canonicalized by the repair script and are
    # added only to the new release.  The old immutable release is untouched.
    forest_items = read_jsonl(args.repair_output / "registries/forest_physical_items.jsonl")
    existing_items = read_jsonl(args.output / "registries/physical_items.jsonl")
    existing_ids = {str(row.get("item_id")) for row in existing_items}
    existing_items.extend(row for row in forest_items if str(row.get("item_id")) not in existing_ids)
    existing_items.sort(key=lambda row: str(row.get("item_id")))
    with (args.output / "registries/physical_items.jsonl").open("w", encoding="utf-8") as handle:
        for row in existing_items:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    existing_frames = read_jsonl(args.output / "registries/frames.jsonl")
    existing_frame_ids = {str(row.get("frame_id")) for row in existing_frames}
    forest_frames = read_jsonl(args.repair_output / "registries/forest_frames.jsonl")
    existing_frames.extend(row for row in forest_frames if str(row.get("frame_id")) not in existing_frame_ids)
    existing_frames.sort(key=lambda row: str(row.get("frame_id")))
    with (args.output / "registries/frames.jsonl").open("w", encoding="utf-8") as handle:
        for row in existing_frames:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    repair_package = read_json(args.repair_output / "REPAIR_PACKAGE.json", {})
    handoff = read_json(args.repair_output / "handoff/model_agent_handoff.json", {})
    view_readiness = read_json(args.repair_output / "audits/view_readiness.json", {})
    comparison_path = args.data_only_comparison or (args.repair_output / "audits/data_only_D0_D3_comparison.json")
    data_only_comparison = read_json(comparison_path, {})
    if args.data_only_comparison is not None:
        copy_file(args.data_only_comparison, args.output / "audits/retrieval_semantic_repair/data_only_D0_D3_comparison.json")
    difficulty_path = args.output / "source_reports/retrieval_semantic_repair/retrieval_data_difficulty_report.json"
    difficulty = read_json(difficulty_path, {})
    if data_only_comparison.get("status") == "PASS_FROZEN_B1_DATA_ONLY_METRICS":
        difficulty["frozen_anchor_baseline"] = {
            "status": data_only_comparison["status"],
            "same_frozen_model": data_only_comparison.get("same_frozen_model"),
            "checkpoint_sha256": data_only_comparison.get("checkpoint_sha256"),
            "feature_cache_sha256": data_only_comparison.get("feature_cache_sha256"),
            "comparison_artifact": "audits/retrieval_semantic_repair/data_only_D0_D3_comparison.json",
            "metrics_by_view": {
                name: value.get("metrics")
                for name, value in (data_only_comparison.get("views") or {}).items()
            },
            "paired_bootstrap": data_only_comparison.get("paired_bootstrap"),
            "improvement_claim": data_only_comparison.get("improvement_claim", False),
        }
        write_json(difficulty_path, difficulty)
    if data_only_comparison.get("status"):
        repair_package = dict(repair_package)
        repair_package["frozen_model_comparison_status"] = data_only_comparison["status"]
    if data_only_comparison.get("status") == "PASS_FROZEN_B1_DATA_ONLY_METRICS":
        handoff = dict(handoff)
        handoff["frozen_model_comparison_status"] = data_only_comparison["status"]
        handoff["data_only_comparison"] = {
            "artifact": "audits/retrieval_semantic_repair/data_only_D0_D3_comparison.json",
            "checkpoint_sha256": data_only_comparison.get("checkpoint_sha256"),
            "feature_cache_sha256": data_only_comparison.get("feature_cache_sha256"),
            "status": data_only_comparison["status"],
            "improvement_claim": data_only_comparison.get("improvement_claim", False),
        }
        write_json(args.output / "handoff/retrieval_semantic_repair/model_agent_handoff.json", handoff)
    physical_count = len(existing_items)
    frame_count = sum(len(row.get("frames", [])) for row in existing_items)
    query_rows = read_jsonl(args.output / "registries/queries.jsonl")
    relevance_graph = write_relevance_graph(args.output / "registries/relevance_graph.jsonl", query_rows)
    # Source-derived verification labels do not replace the identifiability
    # review gate. Report only reviewer-promoted exact rows as verified; this
    # release has zero completed review decisions.
    handoff = dict(handoff)
    handoff["verified_exact_query_count"] = 0
    handoff["relevance_graph"] = {
        "schema_version": relevance_graph["schema_version"],
        "edge_count": relevance_graph["edge_count"],
        "sha256": relevance_graph["sha256"],
    }
    handoff_hashes = dict(handoff.get("hashes") or {})
    handoff_hashes["relevance_graph_sha256"] = relevance_graph["sha256"]
    handoff["hashes"] = handoff_hashes
    comparison_payload = dict(handoff.get("data_only_comparison") or {})
    comparison_payload.update({
        "artifact": "audits/retrieval_semantic_repair/data_only_D0_D3_comparison.json",
        "checkpoint_sha256": data_only_comparison.get("checkpoint_sha256"),
        "feature_cache_sha256": data_only_comparison.get("feature_cache_sha256"),
        "status": data_only_comparison.get("status"),
        "improvement_claim": data_only_comparison.get("improvement_claim", False),
        "gallery_item_count": data_only_comparison.get("gallery_item_count"),
        "metrics_by_view": {
            name: value.get("metrics")
            for name, value in (data_only_comparison.get("views") or {}).items()
        },
        "paired_bootstrap_query_count": (
            data_only_comparison.get("paired_bootstrap", {})
            .get("D0_vs_D1_exact", {})
            .get("paired_query_count")
        ),
        "verified_additions": data_only_comparison.get("verified_additions", {}),
    })
    handoff["data_only_comparison"] = comparison_payload
    view_manifest_hashes = {
        scope: {
            split: {
                "count": len(read_jsonl(args.output / "manifests" / f"{scope}_{split}.jsonl")),
                "sha256": sha256(args.output / "manifests" / f"{scope}_{split}.jsonl"),
            }
            for split in ("train", "development", "test")
        }
        for scope in ("exact", "semantic", "localized", "direction", "stable", "long_series")
    }
    handoff_hashes["generic_no_change_registry_sha256"] = sha256(
        args.output / "registries/generic_no_change_diagnostic.jsonl"
    )
    handoff_hashes["evaluation_sidecars_sha256"] = sha256(
        args.output / "evaluation_sidecars/dense.jsonl"
    )
    handoff["hashes"] = handoff_hashes
    handoff["view_manifest_hashes"] = view_manifest_hashes
    handoff["scope_counts_and_hashes"] = {
        "verified_exact_queries": {
            "count": int(handoff.get("verified_exact_query_count", 0)),
            "candidate_view_manifests": view_manifest_hashes["exact"],
        },
        "semantic_groups": {
            "count": int(handoff.get("semantic_group_count", 0)),
            "sha256": handoff_hashes.get("semantic_groups_sha256"),
        },
        "localized_queries": {
            "count": int(handoff.get("localized_query_count", 0)),
            "view_manifests": view_manifest_hashes["localized"],
            "evaluation_sidecars_sha256": handoff_hashes["evaluation_sidecars_sha256"],
        },
        "stable_queries": {
            "count": int(handoff.get("stable_query_count", 0)),
            "view_manifests": view_manifest_hashes["stable"],
        },
        "direction_queries": {
            "count": int(handoff.get("direction_query_count", 0)),
            "view_manifests": view_manifest_hashes["direction"],
        },
        "long_series_queries": {
            "count": int(handoff.get("long_series_query_count", 0)),
            "view_manifests": view_manifest_hashes["long_series"],
        },
        "disabled_generic_no_change_rows": {
            "count": int(handoff.get("disabled_generic_no_change_row_count", 0)),
            "sha256": handoff_hashes["generic_no_change_registry_sha256"],
        },
        "mask_evaluation_sidecars": {
            "count": int(handoff.get("mask_sidecar_count", 0)),
            "sha256": handoff_hashes["evaluation_sidecars_sha256"],
        },
    }
    write_json(args.output / "handoff/retrieval_semantic_repair/model_agent_handoff.json", handoff)
    release = {
        "schema_version": "qcpr-dataset-v2-retrieval-semantic-repair-v1",
        "release_name": args.release_name,
        "release_state": "DATA_QUALITY_HOLD",
        "branch": args.source_branch,
        "code_sha": git_value(args.repo, "rev-parse", "HEAD"),
        "source_release": str(args.base_release),
        "source_release_sha256s": sha256(args.base_release / "SHA256SUMS"),
        "physical_item_count": physical_count,
        "frame_count": frame_count,
        "query_count": len(query_rows),
        "relevance_graph": relevance_graph,
        "required_manifest_names": REQUIRED_MANIFESTS,
        "manifest_counts": {name: len(read_jsonl(args.output / "manifests" / name)) for name in REQUIRED_MANIFESTS},
        "manifest_hashes_before_release_metadata": {name: sha256(args.output / "manifests" / name) for name in REQUIRED_MANIFESTS},
        "view_status": view_readiness.get("views", {}),
        "training_authorized": False,
        "training_submitted": False,
        "main_training_allowed": False,
        "p1_submitted": False,
        "p2_submitted": False,
        "all_query_training_enabled": all(bool(row.get("training_enabled")) for row in query_rows),
        "all_query_training_disabled": all(not bool(row.get("training_enabled")) for row in query_rows),
        "repair_package": repair_package,
        "data_only_comparison": data_only_comparison,
        "model_agent_handoff": handoff,
        "source_integration": {
            "forest": (
                "PHYSICAL_SCENE_COMPONENT_DISJOINT_MATERIALIZED_TEXT_HOLD"
                if forest_split_audit
                else "PHYSICAL_SCENE_DISJOINT_CANDIDATE_TEXT_HOLD"
            ),
            "tamm": (
                "PHYSICAL_489_SEQUENCE_FIELDS_MATERIALIZED_TEXT_HOLD"
                if tamm_repair_audit and tamm_annotation_source.exists()
                else "PHYSICAL_489_SEQUENCE_TEXT_HOLD"
            ),
            "rscc": "VERIFIED_TEXT_ZERO_HOLD",
            "dubai_cc": "NOT_INTEGRATED",
            "rsrcc": "NOT_INTEGRATED",
            "dynamic_earth_net": "NOT_INTEGRATED",
            "spacenet_7": "NOT_INTEGRATED",
            "terra_cd": "NOT_INTEGRATED",
        },
    }
    write_json(args.output / "RELEASE.json", release)
    checksums = write_checksums(args.output)
    integrity = verify_checksums(args.output)
    write_json(args.output / "audits/retrieval_semantic_repair_integrity.json", {
        "schema_version": "qcpr-retrieval-semantic-repair-integrity-v1",
        "release": str(args.output),
        "physical_item_count": physical_count,
        "frame_count": frame_count,
        "query_count": len(query_rows),
        "relevance_graph": relevance_graph,
        "manifest_count": len(REQUIRED_MANIFESTS),
        "manifest_nonempty": {name: len(read_jsonl(args.output / "manifests" / name)) > 0 for name in REQUIRED_MANIFESTS},
        "checksum_entry_count_before_integrity_audit": len(checksums),
        "checksum_verification_before_integrity_audit": integrity,
        "forest_physical_added": len(forest_items),
        "all_query_training_enabled_false": all(not bool(row.get("training_enabled")) for row in query_rows),
        "exact_gate_passed": False,
        "stable_gate_passed": False,
        "status": (
            "HOLD_REVIEW_REQUIRED"
            if integrity["passed"] and data_only_comparison.get("status") == "PASS_FROZEN_B1_DATA_ONLY_METRICS"
            else "HOLD_REVIEW_AND_MODEL_COMPARISON_REQUIRED"
            if integrity["passed"]
            else "HOLD_CHECKSUM_FAILURE"
        ),
    })
    # The integrity audit is part of the immutable release, so regenerate the
    # checksum file once after writing it and verify the final set.
    write_checksums(args.output)
    final_integrity = verify_checksums(args.output)
    if not final_integrity["passed"]:
        raise SystemExit(json.dumps(final_integrity, sort_keys=True))

    # Refresh the validator report against this exact release.  Copying a
    # baseline validator report is insufficient because its counts and
    # manifest hashes describe a different physical/query registry.
    validator_path = args.output / "audits/validate_qcpr_release_contract.json"
    subprocess.run(
        [
            sys.executable,
            str(args.repo / "scripts/validate_qcpr_release_contract.py"),
            "--release",
            str(args.output),
            "--output",
            str(validator_path),
            "--decode-sample",
            "256",
        ],
        check=True,
    )
    validator = read_json(validator_path, {})
    if not validator.get("passed"):
        raise SystemExit(json.dumps(validator, sort_keys=True))
    pre_validator_checksums = write_checksums(args.output)
    pre_validator_integrity = verify_checksums(args.output)
    if not pre_validator_integrity["passed"]:
        raise SystemExit(json.dumps(pre_validator_integrity, sort_keys=True))
    integrity_path = args.output / "audits/retrieval_semantic_repair_integrity.json"
    integrity_record = read_json(integrity_path, {})
    integrity_record["release_contract_validation"] = {
        "passed": validator["passed"],
        "manifest_count": validator["manifest_count"],
        "physical_item_count": validator["physical_item_count"],
        "frame_count": validator["frame_count"],
        "query_count": validator["query_count"],
        "query_errors": validator["query_errors"],
        "physical_errors": validator["physical_errors"],
        "decode_failures": validator["decode_failures"],
        "mask_free_forbidden_key_hits": validator["mask_free_forbidden_key_hits"],
    }
    if data_only_comparison.get("status") == "PASS_FROZEN_B1_DATA_ONLY_METRICS":
        integrity_record["status"] = "HOLD_REVIEW_REQUIRED"
    integrity_record["checksum_entry_count_before_integrity_audit"] = len(pre_validator_checksums)
    integrity_record["checksum_verification_before_integrity_audit"] = pre_validator_integrity
    integrity_record["final_sha256sums"] = {
        "entries": len(pre_validator_checksums),
        "failures": [],
        "passed": True,
    }
    write_json(integrity_path, integrity_record)
    final_checksums = write_checksums(args.output)
    final_integrity = verify_checksums(args.output)
    if not final_integrity["passed"]:
        raise SystemExit(json.dumps(final_integrity, sort_keys=True))
    write_decision_package(
        args.output,
        branch=args.source_branch,
        code_sha=git_value(args.repo, "rev-parse", "HEAD"),
        release_path=args.output,
        items=existing_items,
        queries=query_rows,
        handoff=handoff,
        view_readiness=view_readiness,
        data_only_comparison=data_only_comparison,
        integrity=final_integrity,
    )
    final_checksums = write_checksums(args.output)
    final_integrity = verify_checksums(args.output)
    if not final_integrity["passed"]:
        raise SystemExit(json.dumps(final_integrity, sort_keys=True))
    print(json.dumps({"release": str(args.output), "physical_items": physical_count, "frames": frame_count, "queries": len(query_rows), "checksum_entries": len(final_checksums), "validator_passed": validator["passed"], "status": "DATA_QUALITY_HOLD"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
