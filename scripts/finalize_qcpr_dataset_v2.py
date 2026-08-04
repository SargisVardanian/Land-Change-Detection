#!/usr/bin/env python3
"""Build the immutable QCPR Dataset-v2 final release.

The command consumes already audited registries and pilot artifacts.  It does
not download data, create captions, fabricate reviews, or launch training.
All policy decisions are explicit command-line inputs and release metadata.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from qcpr_data.contracts.validation import read_jsonl, sha256_file, write_json, write_jsonl
from qcpr_data.manifests.builders import write_release_layout
from qcpr_data.manifests.integrity import audit_release, write_sha256sums
from qcpr_data.identities.physical_graph import build_identity_graph, connected_components
from qcpr_data.queries.exact import build_exact_queries
from qcpr_data.queries.localized import build_localized_eval_queries
from qcpr_data.queries.long_series import build_long_series_queries
from qcpr_data.queries.semantic import build_semantic_eval_queries
from qcpr_data.queries.stable import build_stable_queries
from qcpr_data.reports.decision_package import summarize, write_markdown
from qcpr_data.sources.common import normalize_pair_row, normalize_sequence_row
from qcpr_data.splits.scene_disjoint import stable_holdout_split


VERIFIED = {"human", "human_rewritten", "human_adjudicated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-release", type=Path, required=True)
    parser.add_argument("--pair-registry", type=Path, required=True)
    parser.add_argument("--caption-registry", type=Path, required=True)
    parser.add_argument("--localized-manifest", type=Path, required=True)
    parser.add_argument("--dense-label-registry", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--tamms-manifest", type=Path, required=True)
    parser.add_argument("--tamms-text", type=Path, required=True)
    parser.add_argument("--forest-pilot-dir", type=Path, required=True)
    parser.add_argument("--model-contract-root", type=Path)
    parser.add_argument("--tamms-archive", type=Path)
    parser.add_argument("--tamms-metadata", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def source_name(value: Any) -> str:
    lowered = str(value or "").casefold().replace("_", "-")
    if lowered == "levir-mci":
        return "levir_mci"
    if lowered == "second-cc":
        return "second_cc"
    if lowered in {"rscc-ebd", "rscc_ebd"}:
        return "rscc_ebd"
    if lowered == "s2looking":
        return "s2looking"
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def model_contract_snapshot(root: Path | None) -> dict[str, Any]:
    """Read Model-Agent-owned contract evidence without editing it."""

    if root is None:
        return {"status": "MODEL_CONTRACT_ROOT_NOT_PROVIDED"}
    status_path = root / "contracts/qcpr_shared/model_status.json"
    requirements_path = root / "contracts/qcpr_shared/model_requirements.json"
    request_path = root / "contracts/qcpr_shared/handoff/model_to_dataset.jsonl"
    missing = [str(path) for path in (status_path, requirements_path, request_path) if not path.is_file()]
    if missing:
        return {"status": "MODEL_CONTRACT_INPUT_MISSING", "root": str(root), "missing": missing}
    status = read_json(status_path)
    requirements = read_json(requirements_path)
    requests = read_jsonl(request_path)
    return {
        "status": "MODEL_CONTRACT_AVAILABLE",
        "root": str(root),
        "branch": status.get("branch"),
        "code_sha": status.get("code_sha"),
        "model_status": status.get("status"),
        "dataset_contract_status": status.get("dataset_contract_status"),
        "main_training_allowed": bool(status.get("main_training_allowed")),
        "p2_allowed": bool(status.get("p2_allowed")),
        "requirements_sha256": sha256_file(requirements_path),
        "model_to_dataset_sha256": sha256_file(request_path),
        "open_request_ids": [str(row.get("request_id")) for row in requests if str(row.get("status")) == "OPEN"],
        "required_item_fields": requirements.get("required_fields", {}).get("item", []),
        "required_query_fields": requirements.get("required_fields", {}).get("query", []),
        "query_scopes": requirements.get("query_scopes", []),
        "primary_training_verification": requirements.get("verification_for_primary_training", []),
        "minimum_frames": requirements.get("minimum_frames"),
        "maximum_frames": requirements.get("maximum_frames"),
    }


def tamms_download_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Record acquisition facts and stop conditions without downloading."""

    archive = args.tamms_archive or args.project_root / "datasets/raw/TAMMs/waste_disposal.tar"
    metadata = args.tamms_metadata or args.project_root / "datasets/raw/TAMMs/tamms_data.json"
    archive_exists = archive.is_file()
    metadata_exists = metadata.is_file()
    metadata_count: int | None = None
    if metadata_exists:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        metadata_count = len(value) if isinstance(value, list) else None
    archive_members = 0
    archive_sequences: set[str] = set()
    if archive_exists:
        with tarfile.open(archive) as handle:
            for member in handle:
                archive_members += 1
                parts = Path(member.name).parts
                if len(parts) >= 2:
                    archive_sequences.add("/".join(parts[:2]))
    archive_bytes = archive.stat().st_size if archive_exists else None
    disk_free = shutil.disk_usage(args.project_root).free
    return {
        "schema_version": "qcpr-tamms-download-plan-v1",
        "source": "TAMMs",
        "source_url": "https://huggingface.co/datasets/IceInPot/TAMMs",
        "metadata_path": str(metadata),
        "metadata_exists": metadata_exists,
        "metadata_sequence_count": metadata_count,
        "requested_archive_path": str(archive),
        "current_archive_exists": archive_exists,
        "current_archive_bytes": archive_bytes,
        "current_archive_sha256": sha256_file(archive) if archive_exists else None,
        "current_archive_member_count": archive_members,
        "current_archive_sequence_count": len(archive_sequences),
        "full_archive_expected_bytes": None,
        "full_archive_size_status": "NOT_PUBLISHED_BY_DATASET_CARD",
        "available_disk_bytes_at_plan": disk_free,
        "license_status": "RESEARCH_ONLY_FMOV_TERMS_AND_NONCOMMERCIAL_ANNOTATION_REVIEW_REQUIRED",
        "status": "PILOT_ARCHIVE_ONLY_FULL_ARCHIVE_NOT_ACQUIRED",
        "stop_conditions": [
            "stop before extraction if source terms do not permit the intended use",
            "stop if official archive checksum or source revision cannot be pinned",
            "stop if extracted frame hashes or parent-scene identities are incomplete",
            "stop before training promotion while all captions remain generated_unverified",
        ],
    }


def rscc_ai_audit_status(project_root: Path) -> dict[str, Any]:
    expected_path = project_root / "runs/qcpr_stage2_ai_audit_48.jsonl"
    expected_sha256 = "07863af6c1ee5c38e0e494647f2f29db9db3cdcccf33813f63e1ec6f75b1182e"
    if not expected_path.is_file():
        return {
            "status": "MISSING_INPUT",
            "path": str(expected_path),
            "expected_sha256": expected_sha256,
            "verified": False,
            "advisory": True,
            "human_gate_satisfied": False,
            "training_enabled": False,
        }
    actual = sha256_file(expected_path)
    rows = read_jsonl(expected_path)
    row_ids = {str(row.get("audit_row_id")) for row in rows}
    pair_ids = {str(row.get("canonical_pair_id")) for row in rows}
    return {
        "status": "AVAILABLE_VALIDATED" if actual == expected_sha256 and len(rows) == 48 and len(row_ids) == 48 and len(pair_ids) == 48 else "AVAILABLE_INVALID",
        "path": str(expected_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "row_count": len(rows),
        "unique_audit_row_ids": len(row_ids),
        "unique_canonical_pair_ids": len(pair_ids),
        "verified": actual == expected_sha256 and len(rows) == 48 and len(row_ids) == 48 and len(pair_ids) == 48,
        "advisory": True,
        "human_gate_satisfied": False,
        "training_enabled": False,
    }


def source_balance_audit(
    items: Iterable[Mapping[str, Any]],
    queries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit source/event composition for the only enabled query population."""

    item_rows = {str(row.get("item_id")): row for row in items}
    training_queries = [row for row in queries if bool(row.get("training_enabled"))]
    source_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    event_counts: collections.Counter[str] = collections.Counter()
    for query in training_queries:
        positive_ids = [str(value) for value in query.get("positive_item_ids", [])]
        sources = {str(item_rows[item_id].get("source")) for item_id in positive_ids if item_id in item_rows}
        for source in sorted(sources):
            source_counts[source] += 1
        split_counts[str(query.get("split"))] += 1
        for item_id in positive_ids:
            event_id = item_rows.get(item_id, {}).get("event_id")
            if event_id is not None:
                event_counts[str(event_id)] += 1
    new_sources = sorted(source for source in source_counts if source not in {"levir_mci", "second_cc"})
    event_total = sum(event_counts.values())
    max_event_fraction = max((count / event_total for count in event_counts.values()), default=0.0)
    return {
        "schema_version": "qcpr-source-balance-audit-v1",
        "training_query_count": len(training_queries),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "event_uniform_validation": "NOT_APPLICABLE_NO_TRAINING_EVENT_IDS" if not event_counts else "CHECKED",
        "max_event_fraction": max_event_fraction,
        "max_event_fraction_policy": 0.1,
        "new_source_fraction": len(new_sources) / len(source_counts) if source_counts else 0.0,
        "new_sources": new_sources,
        "max_new_source_fraction_initial_pilot": 0.3,
        "passed": max_event_fraction <= 0.1 and (len(new_sources) / len(source_counts) if source_counts else 0.0) <= 0.3,
    }


def acquired_archive_inventory(project_root: Path, tamms_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    forest_archive = project_root / "datasets/raw/Forest-Change/Forest-Change-dataset.zip"
    if forest_archive.is_file():
        records.append(
            {
                "source": "Forest-Change",
                "path": str(forest_archive),
                "bytes": forest_archive.stat().st_size,
                "sha256": sha256_file(forest_archive),
                "status": "ACQUIRED_PILOT_ARCHIVE",
            }
        )
    records.append(
        {
            "source": "TAMMs",
            "path": tamms_plan["requested_archive_path"],
            "bytes": tamms_plan["current_archive_bytes"],
            "sha256": tamms_plan["current_archive_sha256"],
            "status": tamms_plan["status"],
        }
    )
    return records


def build_pair_items(pair_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    by_source_id: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        raw_source = source_name(row.get("source_dataset") or row.get("dataset_name"))
        if raw_source not in {"levir_mci", "second_cc", "rscc_ebd", "s2looking"}:
            continue
        source_split = str(row.get("split") or "")
        if raw_source == "s2looking" and source_split != "test":
            continue
        if raw_source in {"levir_mci", "second_cc"}:
            split = stable_holdout_split(str(row.get("canonical_pair_id")), source_split=source_split, holdout_fraction=0.2)
            training_enabled = split != "test"
            quality = "VERIFIED_HUMAN_TEXT_AVAILABLE"
            revision = "existing"
        elif raw_source == "rscc_ebd":
            split = source_split if source_split in {"train", "development", "test"} else "train"
            training_enabled = False
            quality = "PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED"
            revision = "stage2-pilot"
        else:
            split = "test"
            training_enabled = False
            quality = "EVALUATION_ONLY_DENSE_SOURCE"
            revision = "existing"
        item = normalize_pair_row(
            row,
            source=raw_source,
            source_revision=revision,
            split=split,
            training_enabled=training_enabled,
            quality_status=quality,
            item_id_value=(str(row.get("canonical_pair_id")) if raw_source != "rscc_ebd" else f"rscc_ebd:{row.get('canonical_pair_id')}"),
            provenance={
                "source_registry_row": str(row.get("canonical_pair_id")),
                "source_split": source_split,
                "split_policy": "preserve_train_and_deterministically_hold_out_source_validation" if raw_source in {"levir_mci", "second_cc"} else "source_split_preserved",
                "source_training_enabled": raw_source in {"levir_mci", "second_cc"},
            },
        ).to_dict()
        items.append(item)
        by_source_id[str(row.get("canonical_pair_id"))] = item
    # The historical SECOND-CC validation partition contains a small number
    # of reversed copies.  Keep both physical rows for compatibility, but
    # collapse their split assignment to the earliest eligible partition so a
    # reversed pair can never cross development/test.
    reverse_groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        frame_signature = tuple(sorted(str(frame.get("sha256") or frame.get("path")) for frame in item.get("frames", [])))
        if len(frame_signature) == 2:
            reverse_groups[frame_signature].append(item)
    split_rank = {"train": 0, "development": 1, "test": 2}
    for signature, group in reverse_groups.items():
        if len(group) < 2 or len({str(item["split"]) for item in group}) < 2:
            continue
        target = min((str(item["split"]) for item in group), key=lambda split: split_rank.get(split, 99))
        for item in group:
            old_split = str(item["split"])
            item["split"] = target
            item["training_enabled"] = bool(item["training_enabled"] and target != "test")
            provenance = dict(item.get("provenance") or {})
            provenance["reversed_pair_component_split_repaired"] = True
            provenance["reversed_pair_component_signature"] = list(signature)
            provenance["reversed_pair_original_split"] = old_split
            provenance["reversed_pair_final_split"] = target
            item["provenance"] = provenance
    return items, by_source_id


def build_tamms_items(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(manifest_path):
        sequence_id = str(row.get("sequence_id"))
        source_split = str(row.get("split") or "train")
        final_split = source_split if source_split in {"train", "development", "test"} else "train"
        item = normalize_sequence_row(
            row,
            source="tamms",
            source_revision=str(row.get("source_version") or "pilot"),
            split=final_split,
            training_enabled=False,
            quality_status="PILOT_PHYSICAL_ONLY_TEXT_HOLD",
            item_id_value=sequence_id,
            provenance={
                "source_sequence_id": row.get("source_sequence_id"),
                "source_split": source_split,
                "split_policy": row.get("split_policy"),
                "generated_text_training_enabled": False,
            },
        ).to_dict()
        if sequence_id in by_id:
            raise ValueError(f"duplicate TAMMs sequence {sequence_id}")
        items.append(item)
        by_id[sequence_id] = item
    return items, by_id


def align_tamms_text(
    rows: Iterable[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join pilot text to physical sequences by stable source sequence ID.

    Archive revisions intentionally change item IDs.  The source sequence path
    is the permitted join key; this avoids silently dropping verified status
    or pretending that a pilot ID is the new physical identity.
    """

    source_sequence_to_item: dict[str, str] = {}
    for item_id, item in items.items():
        source_sequence_id = str(item.get("provenance", {}).get("source_sequence_id") or "")
        if source_sequence_id:
            source_sequence_to_item[source_sequence_id] = str(item_id)
    aligned: list[dict[str, Any]] = []
    for row in rows:
        sequence_id = str(row.get("sequence_id") or "")
        if sequence_id in items:
            aligned.append(dict(row))
            continue
        source_sequence_id = sequence_id.split(":", 2)[-1] if sequence_id.count(":") >= 2 else sequence_id
        target_item = source_sequence_to_item.get(source_sequence_id)
        if target_item is not None:
            aligned.append({**dict(row), "sequence_id": target_item, "source_sequence_id": source_sequence_id})
    return aligned


def audit_asset_hashes(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    missing: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"checked": 0, "missing": 0, "mismatches": 0})
    for item in items:
        source = str(item.get("source"))
        for frame in item.get("frames", []):
            path = Path(str(frame.get("path") or ""))
            if not path.is_file():
                missing.append({"item_id": item.get("item_id"), "frame_id": frame.get("frame_id"), "path": str(path)})
                by_source[source]["missing"] += 1
                continue
            actual = sha256_file(path)
            checked += 1
            by_source[source]["checked"] += 1
            if actual != str(frame.get("sha256")):
                mismatches.append(
                    {
                        "item_id": item.get("item_id"),
                        "frame_id": frame.get("frame_id"),
                        "path": str(path),
                        "expected": frame.get("sha256"),
                        "actual": actual,
                    }
                )
                by_source[source]["mismatches"] += 1
    return {
        "schema_version": "qcpr-physical-asset-hash-audit-v1",
        "frames_checked": checked,
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_examples": missing[:20],
        "mismatch_examples": mismatches[:20],
        "by_source": dict(sorted(by_source.items())),
        "passed": not missing and not mismatches,
    }


def write_global_identity_artifacts(root: Path, items: list[Mapping[str, Any]]) -> dict[str, Any]:
    graph = build_identity_graph(items)
    components = connected_components(graph)
    write_jsonl(root / "audits/global_physical_identity_graph.jsonl", components)
    hash_sources: dict[str, set[str]] = collections.defaultdict(set)
    for item in items:
        for frame in item.get("frames", []):
            hash_sources[str(frame.get("sha256") or frame.get("path"))].add(str(item.get("source")))
    cross_source = sorted((digest, sorted(sources)) for digest, sources in hash_sources.items() if len(sources) > 1)
    write_json(
        root / "audits/cross_source_overlap_audit.json",
        {
            "shared_physical_frame_count": len(cross_source),
            "examples": [{"frame_identity": digest, "sources": sources} for digest, sources in cross_source[:50]],
            "passed": not cross_source,
        },
    )
    component_by_node = {node: component["component_id"] for component in components for node in component["nodes"]}
    summaries: dict[str, dict[str, Any]] = {}
    for item in items:
        source = str(item.get("source"))
        scene_node = f"scene:{source}:{item.get('scene_id')}"
        component_id = component_by_node.get(scene_node, "unresolved")
        entry = summaries.setdefault(component_id, {"component_id": component_id, "item_count": 0, "sources": set()})
        entry["item_count"] += 1
        entry["sources"].add(source)
    with (root / "audits/source_component_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component_id", "item_count", "sources"])
        writer.writeheader()
        for component_id in sorted(summaries):
            entry = summaries[component_id]
            writer.writerow({"component_id": component_id, "item_count": entry["item_count"], "sources": ";".join(sorted(entry["sources"]))})
    return {"component_count": len(components), "cross_source_overlap_count": len(cross_source), "passed": not cross_source}


def source_rows(source_registry_path: Path) -> list[dict[str, Any]]:
    raw_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    if isinstance(raw_registry, list):
        rows = [dict(row) for row in raw_registry]
    elif isinstance(raw_registry, dict):
        rows = [dict(row) for row in raw_registry.get("sources", [])]
    else:
        raise ValueError(f"{source_registry_path} must contain a source list or object")
    names = {str(row.get("source_dataset")) for row in rows}
    additions = {
        "DUBAI-CC": {
            "official_locations": ["https://service.tib.eu/ldmservice/dataset/dubai-cc--a-dataset-for-remote-sensing-change-captioning"],
            "license": "NOT_PINNED",
            "license_status": "REVIEW_REQUIRED",
            "state": "OFFICIAL_LINK_FOUND_ARCHIVE_NOT_ACQUIRED",
            "roles": ["external_exact_evaluation"],
            "blocker": "official archive/access not acquired; one obsolete URL is insufficient to classify source unavailable",
            "physical_pair_count": 500,
            "text_count": 2500,
        },
        "DynamicEarthNet": {
            "official_locations": ["https://mediatum.ub.tum.de/1650201", "https://dataserv.ub.tum.de/index.php/s/m1650201"],
            "license": "CC_BY_SA_4.0",
            "license_status": "CC_BY_SA_4.0_OFFICIAL_RECORD_ARCHIVE_NOT_ACQUIRED",
            "state": "OFFICIAL_ARCHIVE_NOT_ACQUIRED",
            "roles": ["long_series", "land_cover_evaluation"],
            "blocker": "official physical archive not acquired; 75-AOI claim remains source audit context",
            "physical_sequence_count": 75,
            "text_count": None,
        },
        "Forest-Change": {
            "official_locations": ["https://huggingface.co/datasets/JimmyBrocko/Forest-Change"],
            "license": "MIT",
            "license_status": "MIT_DATASET_CARD_WITH_ACADEMIC_REUSE_NOTE",
            "state": "PHYSICAL_READY_TEXT_HOLD",
            "roles": ["temporal_retrieval", "dense_evaluation"],
            "blocker": "scene-disjoint proposal and caption provenance acceptance pending",
            "physical_pair_count": 334,
            "text_count": 1670,
            "training_enabled": False,
        },
        "QAG-360K": {
            "official_locations": ["https://github.com/like413/VisTA", "https://drive.google.com/drive/folders/1EiOJNr8bde7apUQwqoN6cjXWKxIz7QdI?usp=sharing"],
            "license": "CC_BY_NC_4.0_RESEARCH_ONLY_DO_NOT_DISTRIBUTE",
            "license_status": "CC_BY_NC_4.0_MASK_DERIVED_RESEARCH_ONLY",
            "state": "NOT_REQUESTED",
            "roles": ["pretraining_only"],
            "blocker": "metadata-only; parent physical identity and license audit required",
            "physical_pair_count": None,
            "text_count": None,
        },
        "RS5M": {
            "official_locations": [],
            "license": "NOT_PINNED",
            "license_status": "REVIEW_REQUIRED",
            "state": "NOT_REQUESTED",
            "roles": ["pretraining_only"],
            "blocker": "PRETRAINING_ONLY; no temporal retrieval integration",
            "physical_pair_count": None,
            "text_count": None,
        },
        "RSCC-EBD": {
            "official_locations": [],
            "license": "SOURCE_TERMS_RECORDED",
            "license_status": "SOURCE_METADATA_AND_XBD_TERMS_RECORDED_REVIEW_REQUIRED",
            "state": "PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED",
            "roles": ["temporal_retrieval"],
            "blocker": "18,215 physical pairs; verified training text is zero; independent human review required",
            "physical_pair_count": 18215,
            "verified_text_count": 0,
            "text_count": 0,
        },
        "RSRCC": {
            "official_locations": ["https://huggingface.co/datasets/google/RSRCC", "https://github.com/google-research/remote-sensing/"],
            "license": "APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED",
            "license_status": "APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED",
            "state": "METADATA_ACQUIRED_PHYSICAL_ASSET_AUDIT_PENDING",
            "roles": ["localized_evaluation"],
            "blocker": "parent overlap and mask-derived provenance audit incomplete",
            "physical_pair_count": None,
            "text_count": None,
            "training_enabled": False,
        },
        "SpaceNet-7": {
            "official_locations": [],
            "license": "NOT_PINNED",
            "license_status": "REVIEW_REQUIRED",
            "state": "OFFICIAL_ARCHIVE_NOT_ACQUIRED",
            "roles": ["long_series", "urban_development_evaluation"],
            "blocker": "official AWS physical archive not acquired; 101-AOI claim remains source audit context",
            "physical_sequence_count": 101,
            "text_count": None,
        },
        "SkyScript": {
            "official_locations": [],
            "license": "NOT_PINNED",
            "license_status": "REVIEW_REQUIRED",
            "state": "NOT_REQUESTED",
            "roles": ["pretraining_only"],
            "blocker": "PRETRAINING_ONLY; no temporal retrieval integration",
            "physical_pair_count": None,
            "text_count": None,
        },
        "TAMMs": {
            "official_locations": ["https://huggingface.co/datasets/IceInPot/TAMMs"],
            "license": "APACHE_METADATA_FMOV_TERMS",
            "license_status": "APACHE_METADATA_FMOV_TERMS_AND_NONCOMMERCIAL_ANNOTATION_RESTRICTION",
            "state": "PHYSICAL_ONLY_TEXT_HOLD",
            "roles": ["long_series"],
            "blocker": "489 physical sequences acquired from one archive subset; full 37,003-sequence archive not acquired; generated text unverified",
            "physical_sequence_count": 489,
            "frame_count": 1956,
            "text_count": 200,
            "verified_text_count": 0,
            "training_enabled": False,
        },
        "TERRA-CD": {
            "official_locations": ["https://github.com/omkarsoak/TERRA-CD", "https://arxiv.org/abs/2605.14651"],
            "license": "DATA_LICENSE_NOT_PINNED",
            "license_status": "OFFICIAL_REPO_NO_RELEASE_ASSET_DATA_LICENSE_REVIEW_REQUIRED",
            "state": "OFFICIAL_REPO_NO_RELEASE_ASSET",
            "roles": ["semantic_transition_evaluation"],
            "blocker": "official physical archive and temporal contract not acquired",
            "physical_pair_count": None,
            "text_count": None,
        },
        "GlobalGeoTree": {
            "official_locations": [],
            "license": "NOT_PINNED",
            "license_status": "REVIEW_REQUIRED",
            "state": "NOT_REQUESTED",
            "roles": ["specialized_evaluation_only"],
            "blocker": "SPECIALIZED_EVALUATION_ONLY; no temporal retrieval integration",
            "physical_pair_count": None,
            "text_count": None,
        },
    }
    for name, addition in additions.items():
        if name not in names:
            rows.append({"source_dataset": name, "training_enabled": False, **addition})
            continue
        for row in rows:
            if str(row.get("source_dataset")) != name:
                continue
            for key, value in addition.items():
                if key == "official_locations":
                    existing = [str(location) for location in row.get(key, []) or []]
                    row[key] = sorted(set(existing).union(str(location) for location in value or []))
                elif value is not None:
                    row[key] = value
    for row in rows:
        name = str(row.get("source_dataset"))
        if name in {"LEVIR-MCI", "SECOND-CC"}:
            row["license_status"] = "SOURCE_TERMS_RECORDED_DATASET_REDISTRIBUTION_REVIEW_REQUIRED"
            row["source_code_or_repo_license"] = "official repository/source terms; dataset archive terms are separate"
        elif name == "Forest-Change":
            row["license_status"] = "MIT_DATASET_CARD_WITH_ACADEMIC_REUSE_NOTE"
        elif name == "TAMMs":
            row["license_status"] = "APACHE_METADATA_FMOV_TERMS_AND_NONCOMMERCIAL_ANNOTATION_RESTRICTION"
        elif name in {"RSCC-EBD", "RSCC_EBD"}:
            row["license_status"] = "SOURCE_METADATA_AND_XBD_TERMS_RECORDED_REVIEW_REQUIRED"
        else:
            row.setdefault("license_status", "REVIEW_REQUIRED")
    return sorted(rows, key=lambda row: str(row.get("source_dataset")))


def rsrcc_source_audit(project_root: Path) -> dict[str, Any]:
    """Record pinned RSRCC metadata facts without promoting generated text."""

    root = project_root / "datasets/raw/RSRCC"
    metadata_paths = sorted(root.glob("*_metadata.csv"))
    if not metadata_paths:
        return {
            "schema_version": "qcpr-rsrcc-source-audit-v1",
            "status": "MISSING_INPUT",
            "official_location": "https://huggingface.co/datasets/google/RSRCC",
            "revision": "7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea",
            "training_enabled": False,
        }
    pair_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    row_counts: dict[str, int] = {}
    text_count = 0
    for path in metadata_paths:
        split = path.name.removesuffix("_metadata.csv")
        count = 0
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                count += 1
                text_count += bool(str(row.get("text") or "").strip())
                pair = (str(row.get("before_file_name") or ""), str(row.get("after_file_name") or ""))
                if all(pair):
                    pair_splits[pair].add(split)
        row_counts[split] = count
    duplicate_rows = sum(count for count in row_counts.values()) - len(pair_splits)
    cross_split_pairs = sum(1 for splits in pair_splits.values() if len(splits) > 1)
    repository_root = root / "repository"
    image_count = sum(1 for path in repository_root.rglob("*") if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg"}) if repository_root.is_dir() else 0
    return {
        "schema_version": "qcpr-rsrcc-source-audit-v1",
        "status": "METADATA_ACQUIRED_PHYSICAL_ASSET_AUDIT_PENDING" if image_count == 0 else "PHYSICAL_ASSETS_ACQUIRED_PARENT_OVERLAP_AUDIT_PENDING",
        "official_locations": ["https://huggingface.co/datasets/google/RSRCC", "https://github.com/google-research/remote-sensing/"],
        "revision": "7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea",
        "license_status": "APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED",
        "metadata_files": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in metadata_paths
        },
        "row_counts": dict(sorted(row_counts.items())),
        "metadata_row_count": sum(row_counts.values()),
        "unique_physical_pair_filename_tuples": len(pair_splits),
        "duplicate_metadata_rows": duplicate_rows,
        "cross_split_physical_pair_count": cross_split_pairs,
        "text_count": text_count,
        "text_provenance": "generated_language_annotations_from_official_dataset_card; evaluation_only",
        "repository_image_file_count": image_count,
        "parent_overlap_audit": "PENDING_PHYSICAL_IMAGE_ACQUISITION_AND_HASH_COMPARISON",
        "training_enabled": False,
        "download_observation": "Unauthenticated HF physical-asset dry-run encountered HTTP 429; metadata remains the only acquired RSRCC artifact.",
    }


def build_semantic_groups(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        base_group_id = str(row.get("provenance", {}).get("semantic_group_id") or "")
        split = str(row.get("split") or "")
        if not base_group_id:
            continue
        group_id = (base_group_id, split)
        group = grouped.setdefault(
            group_id,
            {
                "group_id": f"{base_group_id}:{split}",
                "source_group_id": base_group_id,
                "split": split,
                "item_ids": [],
                "grades": {},
                "event_ids_are_provenance_only": True,
                "training_enabled": False,
                "evaluation_only": True,
            },
        )
        item_id = str(row["source_item_id"])
        if item_id not in group["item_ids"]:
            group["item_ids"].append(item_id)
        group["grades"][item_id] = max(int(group["grades"].get(item_id, 0)), int(row["graded_relevance"].get(item_id, 0)))
    return sorted(grouped.values(), key=lambda row: row["group_id"])


def copy_forest_hold(root: Path, forest_dir: Path) -> dict[str, Any]:
    target = root / "source_reports/forest"
    target.mkdir(parents=True, exist_ok=True)
    mappings = {
        "forest_physical_registry.jsonl": forest_dir / "forest_change_pair_manifest_mask_free.jsonl",
        "forest_caption_registry.jsonl": forest_dir / "forest_change_text_registry_unverified.jsonl",
        "forest_dense_eval_sidecar.jsonl": forest_dir / "forest_change_dense_label_registry.jsonl",
        "forest_split_integrity.json": forest_dir / "scene_disjoint_split_proposal/forest_change_scene_disjoint_split_proposal.json",
    }
    copied: list[str] = []
    for name, source in mappings.items():
        if source.is_file():
            shutil.copyfile(source, target / name)
            copied.append(name)
    rows = read_jsonl(target / "forest_physical_registry.jsonl") if (target / "forest_physical_registry.jsonl").is_file() else []
    components: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        scene = str(row.get("source_metadata", {}).get("source_scene_group_id") or row.get("pair_id"))
        components[scene].append(str(row.get("pair_id")))
    write_jsonl(target / "forest_scene_components.jsonl", [{"component_id": key, "pair_ids": sorted(value)} for key, value in sorted(components.items())])
    hold = {
        "status": "PHYSICAL_READY_TEXT_HOLD",
        "training_enabled": False,
        "reason": "official pair split has shared image components; proposed scene-disjoint split and caption provenance are not released",
        "copied_artifacts": copied,
        "pair_count": len(rows),
        "component_count": len(components),
    }
    write_json(target / "forest_license_record.json", {"license_status": "MIT_DATASET_CARD_WITH_ACADEMIC_REUSE_NOTE", "redistribution_review_required": True})
    write_json(target / "forest_hold_status.json", hold)
    for split in ("train", "development", "test"):
        write_jsonl(target / f"forest_scene_disjoint_{split}.jsonl", [])
    hashes = {}
    for path in sorted(p for p in target.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        hashes[path.name] = sha256_file(path)
    (target / "SHA256SUMS").write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    return hold


def main() -> int:
    args = parse_args()
    code_sha = git_value(args.repo, "rev-parse", "HEAD")
    branch = git_value(args.repo, "branch", "--show-current")
    release_path = args.output_root / f"qcpr_dataset_v2_final_{code_sha[:7]}_{date.today():%Y%m%d}"
    if release_path.exists():
        raise SystemExit(f"release path already exists: {release_path}")

    model_snapshot = model_contract_snapshot(args.model_contract_root)
    tamms_plan = tamms_download_plan(args)
    ai_audit_status = rscc_ai_audit_status(args.project_root)
    rsrcc_audit = rsrcc_source_audit(args.project_root)
    archive_inventory = acquired_archive_inventory(args.project_root, tamms_plan)

    pair_rows = read_jsonl(args.pair_registry)
    caption_rows = read_jsonl(args.caption_registry)
    items, item_by_source_id = build_pair_items(pair_rows)
    tamms_items, tamms_by_id = build_tamms_items(args.tamms_manifest)
    items.extend(tamms_items)
    item_map = {str(row["item_id"]): row for row in items}
    item_by_source_id.update({key: value for key, value in item_map.items() if key not in item_by_source_id})

    exact_captions = [row for row in caption_rows if source_name(row.get("dataset_name")) in {"levir_mci", "second_cc"}]
    exact = build_exact_queries(exact_captions, item_by_source_id, include_training=True)
    stable = build_stable_queries(exact_captions, item_by_source_id)
    direction = [
        dict(row, query_id=f"{row['query_id']}:direction", query_scope="direction")
        for row in exact
        if row.get("temporal_direction") in {"forward", "reverse"}
    ]

    dense_by_caption = {str(row.get("caption_id")): row for row in read_jsonl(args.dense_label_registry)}
    localized_rows = [row for row in read_jsonl(args.localized_manifest) if str(row.get("split")) == "test"]
    localized, localized_sidecars = build_localized_eval_queries(localized_rows, item_by_source_id, dense_by_caption)

    semantic_source_rows = read_jsonl(args.semantic_manifest)
    semantic = build_semantic_eval_queries(semantic_source_rows, item_by_source_id)
    # The official structured relation source is evaluation-only; preserve the
    # physical item split instead of silently rewriting it to test.

    tamms_text = align_tamms_text(read_jsonl(args.tamms_text), tamms_by_id)
    long_series = build_long_series_queries(tamms_text, tamms_by_id)
    all_query_rows = [*exact, *semantic, *localized, *long_series, *direction, *stable]
    source_balance = source_balance_audit(items, all_query_rows)

    source_registry = source_rows(args.input_release / "registries/source_registry.json")
    forest_status = {
        "status": "PHYSICAL_READY_TEXT_HOLD",
        "training_enabled": False,
        "reason": "official pair split has shared image components; proposed scene-disjoint split and caption provenance are not released",
    }

    exact_train = sum(row.get("split") == "train" for row in exact)
    exact_development = sum(row.get("split") == "development" for row in exact)
    exact_test = sum(row.get("split") == "test" for row in exact)
    statuses = {
        "release_state": "DATA_QUALITY_HOLD",
        "training_authorized": False,
        "p2_real": False,
        "p2_semantic": False,
        "p2_long_series": False,
        "DATASET_EXACT": "READY_EXACT_ONLY_CONDITIONAL_SOURCE_TERMS",
        "DATASET_RSCC": "PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED",
        "DATASET_FOREST": forest_status["status"],
        "DATASET_TAMMS": "PHYSICAL_ONLY_TEXT_HOLD",
        "DATASET_LOCALIZED": "EVAL_ONLY",
        "DATASET_SEMANTIC": "EVAL_ONLY_PRIMARY_GOLD_HOLD",
        "DATASET_LONG_SERIES": "PHYSICAL_ONLY_NO_VALID_QUERY_METADATA_HOLD" if not long_series else "PILOT_GENERATED_TEXT_HOLD",
        "exact_compatibility": {
            "source_pair_counts": {"LEVIR-MCI": 8143, "SECOND-CC": 4701},
            "source_validation_holdout_fraction": 0.2,
            "historical_source_splits_preserved_in_provenance": True,
            "final_counts": {"train_queries": exact_train, "development_queries": exact_development, "test_queries": exact_test},
        },
        "model_contract": model_snapshot["status"],
        "model_contract_evidence": model_snapshot,
        "source_balance": source_balance,
    }
    licenses = {
        "schema_version": "qcpr-license-record-v1",
        "disclaimer": "These records preserve source terms and official links; they do not grant redistribution rights.",
        "sources": source_registry,
        "web_evidence": {
            "LEVIR-MCI": "https://github.com/Chen-Yang-Liu/Change-Agent",
            "SECOND-CC": "https://github.com/ChangeCapsInRS/SecondCC",
            "Forest-Change": "https://huggingface.co/datasets/JimmyBrocko/Forest-Change",
            "TAMMs": "https://huggingface.co/datasets/IceInPot/TAMMs",
            "DynamicEarthNet": "https://mediatum.ub.tum.de/1650201",
            "RSRCC": "https://huggingface.co/datasets/google/RSRCC",
            "QAG-360K": "https://github.com/like413/VisTA",
            "TERRA-CD": "https://github.com/omkarsoak/TERRA-CD",
            "DUBAI-CC": "https://service.tib.eu/ldmservice/dataset/dubai-cc--a-dataset-for-remote-sensing-change-captioning",
        },
    }
    audits = {
        "source_registry": {"path": str(args.input_release / "registries/source_registry.json"), "sha256": sha256_file(args.input_release / "registries/source_registry.json")},
        "input_release": str(args.input_release),
        "source_decision": "existing Stage-2 decision package retained; no historical HOLD directory modified",
        "physical_hash_verification": "performed on final release audit where files are available",
        "human_review": {"RSCC": "not complete", "Forest": "not complete", "TAMMs": "not complete"},
        "model_contract": model_snapshot,
        "rscc_ai_audit": ai_audit_status,
        "rsrcc_source_audit": rsrcc_audit,
        "tamms_download_plan": tamms_plan,
        "acquired_archive_inventory": archive_inventory,
        "source_balance": source_balance,
    }
    release_metadata = {
        "release_name": release_path.name,
        "release_state": "DATA_QUALITY_HOLD",
        "code_sha": code_sha,
        "branch": branch,
        "source_base_commit": "47a628f16aaefb98efa8a20d88f4dbe573dc7d47",
        "source_branch": "codex/qcpr-dataset-v2-stage2",
        "training_submitted": False,
        "p1_submitted": False,
        "p2_submitted": False,
        "view_status": statuses,
        "input_artifacts_are_immutable": True,
        "model_contract": model_snapshot,
        "rscc_ai_audit_status": ai_audit_status["status"],
        "rsrcc_source_audit_status": rsrcc_audit["status"],
        "tamms_download_status": tamms_plan["status"],
        "source_balance": source_balance,
    }
    release_summary = write_release_layout(
        release_path,
        items=items,
        exact=exact,
        semantic=semantic,
        localized=localized,
        long_series=long_series,
        direction=direction,
        stable=stable,
        semantic_groups=build_semantic_groups(semantic),
        source_registry=source_registry,
        sidecars=localized_sidecars,
        licenses=licenses,
        audits=audits,
        release_metadata=release_metadata,
    )
    copy_forest_hold(release_path, args.forest_pilot_dir)
    write_json(release_path / "source_reports/tamms_download_plan.json", tamms_plan)
    write_json(release_path / "source_reports/rsrcc_source_audit.json", rsrcc_audit)
    write_json(
        release_path / "source_reports/official_source_audit.json",
        {
            "schema_version": "qcpr-official-source-audit-v1",
            "sources": source_registry,
            "web_evidence": licenses["web_evidence"],
            "acquired_archives": archive_inventory,
            "blocked_or_deferred": [row for row in source_registry if not row.get("training_enabled", False)],
            "disclaimer": licenses["disclaimer"],
        },
    )
    write_json(release_path / "source_reports/model_contract_snapshot.json", model_snapshot)
    write_json(release_path / "source_reports/acquired_archive_inventory.json", archive_inventory)
    write_json(release_path / "audits/rscc_ai_audit_status.json", ai_audit_status)
    write_json(release_path / "audits/source_balance_audit.json", source_balance)
    asset_hash_audit = audit_asset_hashes(items)
    write_json(release_path / "audits/physical_asset_hash_audit.json", asset_hash_audit)
    if not asset_hash_audit["passed"]:
        raise SystemExit(f"physical asset hash audit failed: {asset_hash_audit}")
    write_global_identity_artifacts(release_path, items)
    first_audit = audit_release(release_path)
    write_sha256sums(release_path)
    final_audit = audit_release(release_path)
    integrity = dict(final_audit)
    integrity["sha256sums"] = final_audit.get("sha256sums")
    summary = summarize(
        code_sha=code_sha,
        branch=branch,
        release_path=release_path,
        source_registry=source_registry,
        items=items,
        queries=all_query_rows,
        integrity=integrity,
        statuses=statuses,
    )
    write_json(release_path / "source_reports/decision_package.json", summary)
    write_markdown(release_path / "source_reports/decision_package.md", summary)
    # The decision package is distributed content, so refresh the file hashes
    # after writing it and verify without mutating any audit file.
    write_sha256sums(release_path)
    checksum_audit = {"sha256sums": __import__("qcpr_data.manifests.integrity", fromlist=["verify_sha256sums"]).verify_sha256sums(release_path)}
    write_json(release_path / "audits/checksum_audit.json", checksum_audit)
    write_sha256sums(release_path)
    print(json.dumps({"release": str(release_path), "code_sha": code_sha, "first_audit": first_audit, "final_audit": final_audit, "summary": release_summary}, sort_keys=True))
    return 0 if final_audit.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
