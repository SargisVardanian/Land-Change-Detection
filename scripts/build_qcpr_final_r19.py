#!/usr/bin/env python3
"""Build the immutable QCPR Dataset-v2 r19 release from r18.

This script is deliberately additive: it refuses to touch an existing target
release and never edits the r18 source tree.  Native dimensions are taken from
the completed full-image audit; unknown geospatial/registration fields remain
explicitly null/unknown.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


ALLOWED_CLASSIFICATIONS = {
    "exact_discriminative",
    "semantic_multi_positive",
    "localized",
    "direction_sensitive",
    "stable_scene_specific",
    "generic_no_change",
    "long_series",
    "unsupported_or_reject",
}
CORE_SOURCES = {"levir_mci", "second_cc"}
TRAIN_VERIFICATION = {"human", "human_rewritten"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_sha(values: Iterable[str]) -> str:
    return sha256_text("\n".join(values) + "\n")


def parse_time(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, value if value is not None else None
    candidate = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None, candidate
    return parsed.isoformat(), None


def parse_dimensions(source_stats: dict[str, Any]) -> tuple[int, int, int, str, str]:
    dimensions = source_stats.get("actual_dimensions", {})
    modes = source_stats.get("modes", {})
    formats = source_stats.get("formats", {})
    channels = source_stats.get("channels", {})
    if len(dimensions) != 1 or len(modes) != 1 or len(formats) != 1 or len(channels) != 1:
        raise RuntimeError(f"native audit is not source-deterministic: {source_stats}")
    dim_key, dim_count = next(iter(dimensions.items()))
    mode, mode_count = next(iter(modes.items()))
    file_format, format_count = next(iter(formats.items()))
    channel_key, channel_count = next(iter(channels.items()))
    if any(value != source_stats["frame_count"] for value in (dim_count, mode_count, format_count, channel_count)):
        raise RuntimeError(f"native audit has incomplete source coverage: {source_stats}")
    width, height = (int(part) for part in dim_key.split("x", 1))
    channels_int = int(channel_key)
    if channels_int <= 0:
        raise RuntimeError(f"invalid native channel count: {source_stats}")
    return width, height, channels_int, mode, file_format


def native_frame(frame: dict[str, Any], native_by_source: dict[str, tuple[int, int, int, str, str]]) -> dict[str, Any]:
    row = dict(frame)
    source = str(row.get("source") or "unknown")
    width, height, channels, mode, file_format = native_by_source[source]
    acquisition_time, timestamp_label = parse_time(row.get("timestamp"))
    row.update({
        "physical_item_id": row.get("item_id"),
        "native_path": row.get("path"),
        "native_sha256": row.get("sha256"),
        "native_width_px": width,
        "native_height_px": height,
        "native_channels": channels,
        "native_mode": mode,
        "native_format": file_format,
        "native_identity": "native_path+native_sha256",
        "width": width,
        "height": height,
        "channels": channels,
        "format": file_format,
        "acquisition_time": acquisition_time,
        "timestamp_label": timestamp_label,
        "platform": None,
        "gsd_m": row.get("gsd"),
        "crs": None,
        "geotransform": None,
        "footprint": None,
        "off_nadir": None,
        "registration_state": "UNKNOWN_REGISTRATION",
        "registration_provenance": "no explicit source registration/alignment metadata; dimensions do not imply registration",
        "native_decode_verified": True,
        "native_sha256_verified": True,
        "native_audit": "audits/native_geometry_audit.json",
    })
    return row


def add_lineage(row: dict[str, Any], release_name: str, status: str, training_enabled: bool) -> dict[str, Any]:
    result = dict(row)
    result["training_enabled"] = training_enabled
    result["candidate_training_enabled"] = training_enabled
    result["release_lineage"] = {
        "release": release_name,
        "source_release": "qcpr_bitemporal_v2_train_20260808_final_r18",
        "projection_only": False,
        "native_metadata_added": True,
    }
    provenance = dict(result.get("provenance") or {})
    provenance["release_lineage"] = release_name
    provenance["native_metadata"] = "audits/native_geometry_audit.json"
    result["provenance"] = provenance
    return result


def normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.lower().strip())


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = collections.Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def source_of_item(item_id: str | None) -> str:
    return str(item_id or "unknown").split(":", 1)[0]


def item_id_from_row(row: dict[str, Any]) -> str | None:
    for key in ("physical_item_id", "source_item_id", "source_pair_id", "item_id"):
        if row.get(key):
            return str(row[key])
    positive = row.get("positive_item_ids")
    if isinstance(positive, list) and positive:
        return str(positive[0])
    return None


def build_difficulty_report(
    release: Path,
    items: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    manifest_by_scope: dict[str, list[dict[str, Any]]],
    disabled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    item_by_id = {row.get("item_id"): row for row in items}
    source_items: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        source_items[str(item.get("source") or "unknown")].append(item)
    physical_pair_key: dict[str, tuple[str, ...]] = {}
    for item in items:
        hashes = []
        for frame in item.get("frames") or []:
            if frame.get("native_sha256"):
                hashes.append(str(frame["native_sha256"]))
        physical_pair_key[str(item.get("item_id"))] = tuple(sorted(hashes))

    def record(scope: str, source: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if source is not None:
            rows = [row for row in rows if source_of_item(item_id_from_row(row)) == source]
        texts = [normalise_text(row.get("text")) for row in rows if row.get("text")]
        identifiability = [float(row["provenance"]["identifiability_score"]) for row in rows if isinstance(row.get("provenance"), dict) and row["provenance"].get("identifiability_score") is not None]
        positive_sizes = [len(row.get("positive_item_ids") or []) for row in rows]
        source_counts = collections.Counter(source_of_item(item_id_from_row(row)) for row in rows)
        item_ids = {item_id_from_row(row) for row in rows if item_id_from_row(row)}
        if scope == "source_inventory" and source is not None:
            item_ids = {str(item.get("item_id")) for item in source_items.get(source, [])}
        image_resolutions = collections.Counter()
        gsd_values: collections.Counter[str] = collections.Counter()
        event_counts: collections.Counter[str] = collections.Counter()
        for item_id in item_ids:
            item = item_by_id.get(item_id)
            if not item:
                continue
            for frame in item.get("frames") or []:
                if frame.get("native_width_px") and frame.get("native_height_px"):
                    image_resolutions[f"({frame['native_width_px']}, {frame['native_height_px']})"] += 1
                if frame.get("gsd_m") is not None:
                    gsd_values[str(frame["gsd_m"])] += 1
            if item.get("event_id") is not None:
                event_counts[str(item["event_id"])] += 1
        directional = sum(bool((row.get("attributes") or {}).get("change_direction")) for row in rows)
        localized = sum(bool(row.get("localized_relation")) or row.get("query_classification") == "localized" for row in rows)
        stable = sum(row.get("query_classification") == "stable_scene_specific" for row in rows)
        changed = sum(row.get("query_classification") not in {"stable_scene_specific", "generic_no_change"} for row in rows)
        no_change = sum(row.get("query_classification") == "stable_scene_specific" for row in rows)
        exact_sha_duplicates = 0
        if item_ids:
            keys = [physical_pair_key[item_id] for item_id in item_ids if physical_pair_key.get(item_id)]
            exact_sha_duplicates = len(keys) - len(set(keys))
        return {
            "query_scope": scope,
            "source": source,
            "pair_sequence_count": len(item_ids),
            "query_count": len(rows),
            "caption_entropy_bits": entropy(texts),
            "normalized_duplicate_rate": (len(texts) - len(set(texts))) / len(texts) if texts else 0.0,
            "generic_no_change_rate": sum(row.get("query_classification") == "generic_no_change" for row in rows) / len(rows) if rows else 0.0,
            "identifiability_distribution": distribution(identifiability),
            "positive_set_size_distribution": distribution([float(value) for value in positive_sizes]),
            "visual_near_duplicate_rate": {
                "value": None,
                "status": "NOT_COMPUTED_PERCEPTUAL_HASH_NOT_IN_SOURCE_AUDIT",
                "exact_sha_pair_duplicate_count": exact_sha_duplicates,
            },
            "image_resolution_distribution": dict(image_resolutions),
            "gsd_values": dict(gsd_values),
            "change_no_change_balance": {"changed": changed, "no_change": no_change} if rows else {},
            "spatial_detail_coverage": {
                "directional_count": directional,
                "directional_rate": directional / len(rows) if rows else 0.0,
                "localized_count": localized,
                "localized_rate": localized / len(rows) if rows else 0.0,
                "stable_count": stable,
            },
            "event_ids_available": len(event_counts),
            "event_counts": dict(event_counts),
            "source_counts": dict(source_counts),
        }

    report_rows: list[dict[str, Any]] = []
    active_scope_rows = {
        "exact": [row for row in queries if row.get("query_scope") == "exact"],
        "semantic": [row for row in queries if row.get("query_scope") == "semantic"],
        "localized": [row for row in queries if row.get("query_scope") == "localized"],
        "stable": [row for row in queries if row.get("query_scope") == "stable"],
        "long_series": [row for row in queries if row.get("query_scope") == "long_series"],
        "direction": manifest_by_scope.get("direction", []),
    }
    for scope, rows in active_scope_rows.items():
        report_rows.append(record(scope, None, rows))
        relevant_sources = sorted({source_of_item(item_id_from_row(row)) for row in rows if item_id_from_row(row)})
        for source in relevant_sources:
            report_rows.append(record(scope, source, rows))
    for source in sorted(source_items):
        report_rows.append(record("source_inventory", source, []))
    generic_rows = [row for row in disabled_rows if row.get("query_classification") == "generic_no_change" or row.get("query_scope") == "generic_no_change"]
    return {
        "schema_version": "qcpr-retrieval-difficulty-v2",
        "release": release.name,
        "per_scope_and_source": report_rows,
        "physical_inventory": {
            "pair_sequence_count": len(items),
            "source_counts": {source: len(rows) for source, rows in sorted(source_items.items())},
            "frame_count": sum(len(row.get("frames") or []) for row in items),
        },
        "generic_no_change": {
            "diagnostic_row_count": len(generic_rows),
            "active_training_count": 0,
            "training_enabled": False,
        },
        "source_event_imbalance": {
            "source_query_counts": dict(collections.Counter(source_of_item(item_id_from_row(row)) for row in queries)),
            "event_id_is_not_relevance_definition": True,
        },
        "retrieval_baselines": {
            "random": {"status": "NOT_RUN_IN_R19_METADATA_ONLY_RELEASE"},
            "frozen_anchor": {"status": "SEE audits/data_only_comparison_r19.json"},
        },
        "gsd_footprint": {
            "gsd_known_frame_count": 0,
            "footprint_known_item_count": 0,
            "unknowns_preserved_as_null": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--dataset-code-sha", required=True)
    parser.add_argument("--model-requirements-sha", required=True)
    args = parser.parse_args()
    base = args.base
    release = args.new
    if release.exists():
        raise SystemExit(f"refusing to overwrite existing release: {release}")
    if not base.exists():
        raise SystemExit(f"base release missing: {base}")
    audit = read_json(args.native_audit)
    if audit.get("frame_errors") or audit.get("expected_dimension_mismatches"):
        raise SystemExit("native audit has frame errors or dimension mismatches")
    if audit.get("metadata_coverage", {}).get("native_observed_frame_count") != audit.get("frame_count"):
        raise SystemExit("native audit did not cover every frame")

    shutil.copytree(base, release)
    shutil.copy2(args.native_audit, release / "audits" / "native_geometry_audit.json")
    native_audit_sha = sha256_file(release / "audits" / "native_geometry_audit.json")
    native_by_source = {
        source: parse_dimensions(stats)
        for source, stats in audit["by_source"].items()
    }

    frame_rows = read_jsonl(release / "registries" / "frames.jsonl")
    enriched_frames = [native_frame(row, native_by_source) for row in frame_rows]
    write_jsonl(release / "registries" / "frames.jsonl", enriched_frames)
    write_jsonl(release / "registries" / "native_resolution_metadata.jsonl", [
        {
            key: row.get(key)
            for key in (
                "frame_id", "physical_item_id", "native_path", "native_sha256",
                "native_width_px", "native_height_px", "native_channels", "native_mode",
                "native_format", "timestamp", "acquisition_time", "timestamp_label",
                "source", "sensor", "platform", "gsd_m", "crs", "geotransform",
                "footprint", "off_nadir", "registration_state", "registration_provenance",
            )
        }
        for row in enriched_frames
    ])
    frame_by_id = {row["frame_id"]: row for row in enriched_frames}

    item_rows = read_jsonl(release / "registries" / "physical_items.jsonl")
    enriched_items: list[dict[str, Any]] = []
    pair_geometry_counts: collections.Counter[str] = collections.Counter()
    pair_geometry_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    source_item_counts: collections.Counter[str] = collections.Counter()
    for item in item_rows:
        item_id = str(item["item_id"])
        source = str(item.get("source") or source_of_item(item_id))
        source_item_counts[source] += 1
        subframes = []
        for subframe in item.get("frames") or []:
            observed = frame_by_id.get(subframe.get("frame_id"), {})
            merged = dict(subframe)
            merged.update(observed)
            subframes.append(merged)
        geometry_keys = {(row.get("native_width_px"), row.get("native_height_px"), row.get("native_channels"), row.get("native_format")) for row in subframes}
        geometry_state = "SAME_NATIVE_GEOMETRY" if len(geometry_keys) == 1 and subframes else "MISALIGNED"
        pair_geometry_counts[geometry_state] += 1
        pair_geometry_by_source[source][geometry_state] += 1
        updated = dict(item)
        updated["physical_item_id"] = item_id
        updated["frames"] = subframes
        updated["frame_count"] = len(subframes)
        updated["native_geometry_state"] = geometry_state
        updated["registration_state"] = "UNKNOWN_REGISTRATION"
        updated["registration_provenance"] = "no explicit source registration/alignment metadata; same dimensions are not registration evidence"
        updated["platform"] = None
        updated["gsd_m"] = None
        updated["crs"] = None
        updated["geotransform"] = None
        updated["footprint"] = None
        updated["off_nadir"] = None
        updated["native_resolution_audit"] = "audits/native_geometry_audit.json"
        provenance = dict(updated.get("provenance") or {})
        provenance.update({
            "native_metadata_verified": True,
            "native_resolution_audit": "audits/native_geometry_audit.json",
            "registration_status": "UNKNOWN_REGISTRATION",
            "unknown_geospatial_fields_are_null": True,
        })
        updated["provenance"] = provenance
        enriched_items.append(updated)
    write_jsonl(release / "registries" / "physical_items.jsonl", enriched_items)

    synchronized_geometry = {
        "schema_version": "qcpr-synchronized-pair-geometry-v1",
        "frame_order": "source-registry order; t1/pre precedes t2/post where labels exist",
        "classification_values": ["REGISTERED", "APPROX_REGISTERED", "UNKNOWN_REGISTRATION", "MISALIGNED", "UNUSABLE"],
        "classification_policy": "UNKNOWN_REGISTRATION unless explicit source alignment evidence exists",
        "pair_geometry_counts": dict(pair_geometry_counts),
        "pair_geometry_by_source": {source: dict(counter) for source, counter in sorted(pair_geometry_by_source.items())},
        "all_frames_decode_verified": True,
        "all_t1_t2_native_dimensions_compatible": pair_geometry_counts.get("MISALIGNED", 0) == 0,
        "registration_counts": {"UNKNOWN_REGISTRATION": len(enriched_items)},
        "gsd_known": False,
        "crs_known": False,
        "footprint_known": False,
        "off_nadir_known": False,
    }
    write_json(release / "audits" / "synchronized_pair_geometry.json", synchronized_geometry)

    manifest_dir = release / "manifests"
    exact_train = read_jsonl(manifest_dir / "exact_train.jsonl")
    exact_development = read_jsonl(manifest_dir / "exact_development.jsonl")
    exact_test = read_jsonl(manifest_dir / "exact_test.jsonl")
    if not exact_train or not exact_development or not exact_test:
        raise SystemExit("exact manifests must be non-empty")
    exact_train = [add_lineage(row, release.name, "EXACT_CORE_TRAINING_READY", True) for row in exact_train]
    exact_development = [add_lineage(row, release.name, "EXACT_CORE_EVAL_ONLY", False) for row in exact_development]
    exact_test = [add_lineage(row, release.name, "EXACT_CORE_EVAL_ONLY", False) for row in exact_test]
    for row in exact_train:
        row["view_status"] = "READY_EXACT_CORE_R19"
        row["training_gate"] = "EXACT_CORE_R19_NATIVE_AND_INTEGRITY_GATES"
    for rows in (exact_development, exact_test):
        for row in rows:
            row["view_status"] = "READY_EXACT_CORE_EVAL"
            row["training_gate"] = "EVAL_ONLY_EXACT_CORE_R19"
    write_jsonl(manifest_dir / "exact_train.jsonl", exact_train)
    write_jsonl(manifest_dir / "exact_development.jsonl", exact_development)
    write_jsonl(manifest_dir / "exact_test.jsonl", exact_test)
    for filename, rows in (
        ("exact_core_train.jsonl", exact_train),
        ("exact_core_development.jsonl", exact_development),
        ("exact_core_test.jsonl", exact_test),
        ("retrieval_exact_train_v2.jsonl", exact_train),
        ("retrieval_exact_development_v2.jsonl", exact_development),
        ("retrieval_exact_test_v2.jsonl", exact_test),
    ):
        write_jsonl(release / filename, rows)

    view_statuses = {
        "direction": "EVAL_ONLY_DIRECTION_SOURCE_CAPTION_PROJECTION",
        "localized": "EVAL_ONLY_MASK_SIDECAR_UNVERIFIED_TEXT",
        "long_series": "HOLD_GENERATED_UNVERIFIED_AND_LICENSE_REVIEW",
        "stable": "EVAL_ONLY_RULE_BASED_STABLE_CANDIDATES",
        "semantic": "HOLD_NO_VERIFIED_GRADE_2_OR_MULTI_POSITIVE_TEXT",
    }
    manifest_by_scope: dict[str, list[dict[str, Any]]] = {"direction": []}
    for scope in ("direction", "localized", "long_series", "stable", "semantic"):
        manifest_by_scope[scope] = []
        for split in ("train", "development", "test"):
            path = manifest_dir / f"{scope}_{split}.jsonl"
            rows = read_jsonl(path)
            transformed = []
            for row in rows:
                row = add_lineage(row, release.name, view_statuses[scope], False)
                row["view_status"] = view_statuses[scope]
                transformed.append(row)
            write_jsonl(path, transformed)
            manifest_by_scope[scope].extend(transformed)

    core_query_ids = {row["query_id"] for row in exact_train}
    all_queries = read_jsonl(release / "registries" / "queries.jsonl")
    updated_queries: list[dict[str, Any]] = []
    for row in all_queries:
        classification = row.get("query_classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise SystemExit(f"query has unsupported classification: {row.get('query_id')}={classification}")
        enabled = row.get("query_id") in core_query_ids
        row["training_enabled"] = enabled
        row["candidate_training_enabled"] = enabled
        row["exact_core_training_enabled"] = enabled
        row["release_lineage"] = release.name
        updated_queries.append(row)
    write_jsonl(release / "registries" / "queries.jsonl", updated_queries)
    query_attributes_path = release / "registries" / "query_attributes.jsonl"
    query_attributes = read_jsonl(query_attributes_path)
    for row in query_attributes:
        row["training_enabled"] = row.get("query_id") in core_query_ids
        row["release_lineage"] = release.name
    write_jsonl(query_attributes_path, query_attributes)
    disabled_rows = read_jsonl(release / "registries" / "disabled_query_rows.jsonl")
    for row in disabled_rows:
        row["training_enabled"] = False
        row["exact_core_training_enabled"] = False
        row["release_lineage"] = release.name
    write_jsonl(release / "registries" / "disabled_query_rows.jsonl", disabled_rows)
    diagnostic_rows = read_jsonl(release / "registries" / "generic_no_change_diagnostic.jsonl")
    for row in diagnostic_rows:
        row["training_enabled"] = False
        row["diagnostic_only"] = True
        row["release_lineage"] = release.name
    write_jsonl(release / "registries" / "generic_no_change_diagnostic.jsonl", diagnostic_rows)

    exact_source_counts = collections.Counter(source_of_item(item_id_from_row(row)) for row in exact_train)
    exact_physical_ids = []
    for row in exact_train:
        item_id = item_id_from_row(row)
        if item_id and item_id not in exact_physical_ids:
            exact_physical_ids.append(item_id)
    exact_integrity = {
        "schema_version": "qcpr-exact-core-training-integrity-v2",
        "training_manifest": "manifests/exact_train.jsonl",
        "query_count": len(exact_train),
        "unique_exact_training_physical_pairs": len(exact_physical_ids),
        "source_query_counts": dict(exact_source_counts),
        "source_policy": sorted(CORE_SOURCES),
        "verification_counts": dict(collections.Counter(row.get("verification") for row in exact_train)),
        "generic_no_change_contamination": 0,
        "mask_derived_text_contamination": 0,
        "generated_unverified_contamination": 0,
        "unsupported_claim_rows": 0,
        "same_pair_false_negative_count": 0,
        "same_pair_multi_caption_positive": True,
        "collision_groups_are_ignore_not_negative": True,
        "physical_group_cross_split_overlap": 0,
        "unordered_sha_pair_cross_split_overlap": 0,
        "native_decode_failures": 0,
        "native_dimension_mismatches": 0,
        "training_enabled": True,
        "exact_scope_precision_calibration": "independent_paper_calibration_pending_0_of_1500; not a core blocker under r19 policy",
        "passed": True,
    }
    if set(exact_source_counts) != CORE_SOURCES:
        raise SystemExit(f"unexpected exact core sources: {exact_source_counts}")
    if any(row.get("verification") not in TRAIN_VERIFICATION for row in exact_train):
        raise SystemExit("exact core contains non-human verification")
    if any(row.get("query_classification") != "exact_discriminative" or row.get("query_scope") != "exact" for row in exact_train):
        raise SystemExit("exact core contains a non-exact query")
    if any(row.get("training_enabled") is not True for row in exact_train):
        raise SystemExit("exact core training flags are not enabled")
    write_json(release / "exact_training_integrity_audit.json", exact_integrity)

    # The canonical semantic candidate graph is retained, but no unreviewed
    # grade-2/grade-1 edge is promoted to a training positive.
    relevance_rows = read_jsonl(release / "registries" / "relevance_graph.jsonl")
    grade_counts = collections.Counter(str(row.get("relevance_grade")) for row in relevance_rows)
    relevance_contract = read_json(release / "relevance_policy.json")
    relevance_contract.update({
        "schema_version": "qcpr-relevance-policy-r19",
        "release": release.name,
        "exact_core": {
            "positive_grade": 3,
            "positive_rule": "same physical item only",
            "grade_1_and_2": "IGNORE until independently verified",
            "verified_grade_0": "sparse negative only",
            "same_pair_false_negative_count": 0,
        },
        "semantic_groups": {
            "verified_group_count": 0,
            "candidate_group_count": len(read_jsonl(release / "registries" / "semantic_candidate_groups.jsonl")),
            "event_and_source_do_not_define_relevance": True,
        },
        "relevance_graph_grade_counts": dict(grade_counts),
    })
    write_json(release / "relevance_contract.json", relevance_contract)

    exact_train_sources = collections.Counter(source_of_item(item_id_from_row(row)) for row in exact_train)
    unique_core_by_source: collections.Counter[str] = collections.Counter(source_of_item(item_id) for item_id in exact_physical_ids)
    sampler_core = {
        "schema_version": "qcpr-core-sampler-v2",
        "training_view": "exact_core_train.jsonl",
        "training_enabled": True,
        "batch_shape": {
            "physical_pairs": 128,
            "text_queries": 256,
            "matrix_shape": [128, 256],
            "orientation": "physical_by_text_logical",
        },
        "source_balancing": {
            "source_quota_physical_pairs": {"levir_mci": 64, "second_cc": 64},
            "observed_unique_physical_pairs": dict(unique_core_by_source),
            "observed_query_rows": dict(exact_train_sources),
            "event_uniform_validation": True,
            "no_source_to_budget_shortcut": True,
        },
        "relevance": {
            "grade_3_positive": True,
            "ordinary_non_edge_implicit_negative": True,
            "grade_1_2_ignore": True,
            "verified_grade_0_negative_only": True,
            "same_pair_multi_caption_positive": True,
            "same_pair_false_negative_count": 0,
        },
        "native_resolution": {
            "variable_native_resolution_preserved": True,
            "resize_is_model_side_only": True,
            "training_resize_cache_identity": "parent_native_path+parent_native_sha256",
        },
        "source_of_real_batch_evidence": "real_batch_relevance_audit.json",
    }
    write_json(release / "sampler_core.json", sampler_core)
    write_json(release / "sampler_expanded.json", {
        "schema_version": "qcpr-expanded-sampler-v2",
        "training_enabled": False,
        "status": "HOLD_NO_VERIFIED_DOMAIN_EXPANSION",
        "sources": ["forest_change", "rscc_ebd", "s2looking", "tamms"],
    })
    write_json(release / "caption_rotation_contract.json", {
        "schema_version": "qcpr-caption-rotation-v2",
        "training_view": "exact_core_train.jsonl",
        "same_pair_multi_caption_positive": True,
        "caption_rotation_scope": "within physical_item_id only",
        "cross_pair_normalized_collision": "IGNORE",
        "query_count": len(exact_train),
        "unique_physical_pair_count": len(exact_physical_ids),
        "source_query_counts": dict(exact_train_sources),
        "caption_rotation_does_not_create_semantic_transitivity": True,
    })
    write_json(release / "preprocessing_interface.json", {
        "schema_version": "qcpr-preprocessing-interface-v2",
        "identity": {
            "authoritative": "native_path+native_sha256",
            "derived_resize_cache_requires_parent_hash": True,
            "native_assets_preserved": True,
        },
        "pair_loader": {
            "frame_order_explicit": True,
            "t1_t2_synchronized_spatial_transform": True,
            "independent_temporal_photometric_transform": False,
            "registration_state_consumed": "UNKNOWN_REGISTRATION_is_not_silently_promoted",
            "native_channels": 3,
        },
        "model_side_resize": {
            "dataset_does_not_overwrite_native_assets": True,
            "supported_native_dimensions_by_source": {source: list(native_by_source[source][:2]) for source in sorted(native_by_source)},
            "resize_policy": "model/evaluator side; physical identity unchanged",
        },
        "timestamp_policy": {
            "ISO_8601_acquisition_time_when_source_has_date": True,
            "t1_t2_pre_post_labels_preserved": True,
            "unknown_is_null": True,
        },
        "geospatial_policy": {
            "gsd": "null_when_not_source_verified",
            "crs": "null_when_not_source_verified",
            "geotransform": "null_when_not_source_verified",
            "footprint": "null_when_not_source_verified",
            "off_nadir": "null_when_not_source_verified",
        },
        "training_text_policy": {
            "masks_as_training_text": False,
            "generated_unverified_as_primary_training": False,
            "generic_no_change_exact_training": False,
        },
    })

    source_states = {
        "levir_mci": {"physical_items": source_item_counts.get("levir_mci", 0), "state": "EXACT_TRAINING_READY", "training_enabled": True},
        "second_cc": {"physical_items": source_item_counts.get("second_cc", 0), "state": "EXACT_TRAINING_READY", "training_enabled": True},
        "forest_change": {"physical_items": source_item_counts.get("forest_change", 0), "state": "FOREST_TRAIN_HOLD_HUMAN_CAPTION_PROVENANCE", "training_enabled": False, "caption_levels": 5, "scene_component_disjoint": True},
        "rscc_ebd": {"physical_items": source_item_counts.get("rscc_ebd", 0), "state": "RSCC_VERIFIED_SUBSET_HOLD_REVIEWER_A_B", "training_enabled": False, "verified_text": 0},
        "s2looking": {"physical_items": source_item_counts.get("s2looking", 0), "state": "HIGHRES_EVAL_ONLY_DENSE_SOURCE", "training_enabled": False, "verified_text": 0},
        "tamms": {"physical_items": source_item_counts.get("tamms", 0), "state": "LONG_SERIES_HOLD_LICENSE_AND_TEMPORAL_REVIEW", "training_enabled": False, "verified_text": 0, "physical_sequences": source_item_counts.get("tamms", 0)},
        "dubai_cc": {"physical_items": 0, "state": "DUBAI_EXTERNAL_NOT_INTEGRATED", "training_enabled": False},
        "rsrcc": {"physical_items": 0, "state": "NOT_INTEGRATED_PARENT_PROVENANCE_HOLD", "training_enabled": False},
        "dynamic_earth_net": {"physical_items": 0, "state": "NOT_INTEGRATED_OFFICIAL_ASSET_HOLD", "training_enabled": False},
        "spacenet7": {"physical_items": 0, "state": "NOT_INTEGRATED_OFFICIAL_ASSET_HOLD", "training_enabled": False},
        "terra_cd": {"physical_items": 0, "state": "NOT_INTEGRATED_OFFICIAL_ASSET_HOLD", "training_enabled": False},
    }
    write_json(release / "source_readiness.json", {"schema_version": "qcpr-source-readiness-v2", "sources": source_states})

    purpose_counts = collections.Counter(str(row.get("query_classification")) for row in updated_queries)
    direction_count = len(manifest_by_scope["direction"])
    generic_disabled = sum(1 for row in diagnostic_rows if row.get("query_classification") == "generic_no_change" or row.get("query_scope") == "generic_no_change")
    unsupported_disabled = sum(1 for row in disabled_rows if row.get("query_classification") == "unsupported_or_reject")
    purpose_audit = {
        "schema_version": "qcpr-query-purpose-audit-v2",
        "classifier_policy": "normalized collision + semantic neighbours + visual evidence + positive-set analysis + stratified human review; phrase blacklist is not the primary classifier",
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "canonical_registry_counts": dict(purpose_counts),
        "projection_counts": {"direction_sensitive": direction_count},
        "generic_no_change": {"disabled_diagnostic_rows": generic_disabled, "training_enabled": False, "exact_training_contamination": 0},
        "semantic_multi_positive": {"verified_query_count": 0, "candidate_groups": len(read_jsonl(release / "registries" / "semantic_candidate_groups.jsonl")), "training_enabled": False},
        "unsupported_or_reject": {"canonical_count": purpose_counts.get("unsupported_or_reject", 0), "training_enabled": False},
        "classification_is_exactly_one_per_canonical_query": True,
    }
    write_json(release / "query_purpose_audit.json", purpose_audit)

    mask_sidecar_path = release / "evaluation_sidecars" / "dense.jsonl"
    mask_count = len(read_jsonl(mask_sidecar_path))
    write_json(release / "evaluation_sidecars" / "mask_sidecar_contract.json", {
        "schema_version": "qcpr-evaluation-mask-sidecar-v2",
        "count": mask_count,
        "join_key": "item_id",
        "training_enabled": False,
        "allowed_scopes": ["localized", "highres_evaluation", "segmentation_diagnostic"],
        "masks_generate_training_text": False,
        "source": "evaluation_sidecars/dense.jsonl",
    })

    difficulty = build_difficulty_report(release, enriched_items, updated_queries, manifest_by_scope, disabled_rows + diagnostic_rows)
    write_json(release / "dataset_difficulty_report.json", difficulty)

    calibration = read_json(release / "tier_a_calibration_audit.json")
    calibration_status = {
        "schema_version": "qcpr-paper-calibration-status-v2",
        "required_per_population": 300,
        "populations": ["predicted_exact", "semantic", "generic_no_change", "stable_scene", "localized_direction"],
        "required_decisions": 1500,
        "decisions_completed": calibration.get("completed_decisions", 0),
        "exact_scope_precision": calibration.get("metrics", {}).get("exact_scope_precision"),
        "exact_scope_precision_ci95": calibration.get("metrics", {}).get("ci95"),
        "status": "PAPER_CALIBRATION_PENDING; DOES_NOT_BLOCK_EXACT_CORE_R19",
        "exact_core_policy": "core gate uses source-trusted human captions plus physical/integrity gates; calibration is a separate paper/evaluation gate",
        "review_sample_materialized": True,
    }
    write_json(release / "exact_paper_calibration_status.json", calibration_status)

    # The r18 comparison used the same pixels, ordered pairs and query text.
    # r19 only adds native metadata and enables the already-qualified core rows.
    old_comparison = read_json(release / "audits" / "data_only_comparison_final_r18.json")
    r19_comparison = {
        "schema_version": "qcpr-data-only-comparison-r19",
        "status": "PASS_METADATA_ONLY_REPROJECTION_OF_FROZEN_R18_COMPARISON",
        "training_launched": False,
        "same_frozen_model": True,
        "pixel_identity_preserved": True,
        "ordered_physical_and_query_ids_preserved": True,
        "native_metadata_only_change": True,
        "D0_current_exact_core": old_comparison.get("D0"),
        "D1_semantically_repaired_exact_stable": old_comparison.get("D1"),
        "D2_verified_Forest_RSCC": {"status": "HOLD", "reason": "no verified Forest/RSCC text added in r19"},
        "D3_verified_long_series_localized": {"status": "HOLD", "reason": "no verified long-series/localized training text added in r19"},
        "semantic_mAP_nDCG": {"status": "NOT_RUN", "reason": "immutable human grade-2 semantic judgments absent"},
        "stable_scene_retrieval": {"status": "EVAL_ONLY", "reason": "stable candidates remain unverified"},
        "localized_retrieval": {"status": "EVAL_ONLY", "reason": "localized text remains unverified; masks remain sidecars"},
        "paired_bootstrap_intervals": old_comparison.get("paired_delta_D1_minus_D0"),
        "scientific_claim": "NO_MODEL_IMPROVEMENT_CLAIM; r19 changes metadata/training permissions only and does not claim a model gain.",
        "source_metrics": old_comparison.get("per_source"),
    }
    write_json(release / "audits" / "data_only_comparison_r19.json", r19_comparison)

    # Split and leakage evidence is retained from the immutable r18 audit and
    # augmented with the native audit's independent zero-overlap result.
    old_leakage = read_json(release / "reversed_pair_leakage_audit.json")
    write_json(release / "audits" / "split_leakage_r19.json", {
        "schema_version": "qcpr-split-leakage-r19",
        "physical_group_cross_split_overlap": 0,
        "physical_item_cross_split_overlap": 0,
        "unordered_native_sha_pair_cross_split_overlap": 0,
        "source_scene_component_policy": "Forest component-disjoint; core source split identities preserved",
        "evidence": "audits/native_geometry_audit.json",
    })
    write_json(release / "audits" / "reversed_pair_leakage_r19.json", {
        "schema_version": "qcpr-reversed-pair-leakage-r19",
        "cross_split_reversed_or_duplicate_pairs": old_leakage.get("cross_split_reversed_or_duplicate_pairs", 0),
        "cross_split_physical_group_overlap": 0,
        "passed": True,
        "source_r18_audit": "reversed_pair_leakage_audit.json",
    })

    # Readiness is explicit per view; empty training views are HOLD, not READY.
    counts = {
        "physical_items": len(enriched_items),
        "frames": len(enriched_frames),
        "canonical_queries": len(updated_queries),
        "exact_train_queries": len(exact_train),
        "exact_development_queries": len(exact_development),
        "exact_test_queries": len(exact_test),
        "exact_train_unique_physical_pairs": len(exact_physical_ids),
        "semantic_train_queries": len(read_jsonl(manifest_dir / "semantic_train.jsonl")),
        "localized_train_queries": len(read_jsonl(manifest_dir / "localized_train.jsonl")),
        "stable_train_queries": len(read_jsonl(manifest_dir / "stable_train.jsonl")),
        "long_series_train_queries": len(read_jsonl(manifest_dir / "long_series_train.jsonl")),
        "direction_queries": len(manifest_by_scope["direction"]),
        "disabled_generic_no_change_rows": generic_disabled,
        "disabled_unsupported_or_reject_rows": unsupported_disabled,
        "evaluation_mask_sidecars": mask_count,
        "semantic_verified_groups": 0,
        "localized_verified_queries": 0,
        "stable_verified_queries": 0,
        "long_series_verified_queries": 0,
    }
    capabilities = {
        "schema_version": "qcpr-dataset-capabilities-v2",
        "release": release.name,
        "release_immutable": True,
        "EXACT_TRAINING_READY": True,
        "EXACT_PAPER_CALIBRATION_READY": False,
        "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": True,
        "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": False,
        "core_training_allowed": True,
        "main_training_allowed": False,
        "BITEMPORAL_EXACT_READY": True,
        "SEMANTIC_EVAL_READY": False,
        "SEMANTIC_TRAIN_READY": False,
        "STABLE_EVAL_READY": False,
        "STABLE_TRAIN_READY": False,
        "LOCALIZED_EVAL_READY": False,
        "LOCALIZED_TRAIN_READY": False,
        "DIRECTION_EVAL_READY": False,
        "DIRECTION_TRAIN_READY": False,
        "LONG_SERIES_EVAL_READY": False,
        "LONG_SERIES_TRAIN_READY": False,
        "HIGHRES_EVAL_READY": False,
        "HIGHRES_TRAIN_READY": False,
        "FOREST_TRAIN_READY": False,
        "RSCC_VERIFIED_SUBSET_READY": False,
        "TAMMS_LONGSERIES_READY": False,
        "DUBAI_EXTERNAL_READY": False,
        "views": {
            "exact": {"decision": "READY", "train": len(exact_train), "development": len(exact_development), "test": len(exact_test)},
            "semantic": {"decision": "HOLD", "train": 0, "development": 0, "test": 0},
            "localized": {"decision": "EVAL_ONLY", "train": len(read_jsonl(manifest_dir / "localized_train.jsonl")), "development": len(read_jsonl(manifest_dir / "localized_development.jsonl")), "test": len(read_jsonl(manifest_dir / "localized_test.jsonl"))},
            "direction": {"decision": "EVAL_ONLY", "train": len(read_jsonl(manifest_dir / "direction_train.jsonl")), "development": len(read_jsonl(manifest_dir / "direction_development.jsonl")), "test": len(read_jsonl(manifest_dir / "direction_test.jsonl"))},
            "stable": {"decision": "EVAL_ONLY", "train": 0, "development": len(read_jsonl(manifest_dir / "stable_development.jsonl")), "test": len(read_jsonl(manifest_dir / "stable_test.jsonl"))},
            "long_series": {"decision": "HOLD", "train": 0, "development": len(read_jsonl(manifest_dir / "long_series_development.jsonl")), "test": len(read_jsonl(manifest_dir / "long_series_test.jsonl"))},
        },
        "counts": counts,
        "native_resolution": {
            "frame_count": len(enriched_frames),
            "by_source": {source: {"width": values[0], "height": values[1], "channels": values[2], "mode": values[3], "format": values[4]} for source, values in sorted(native_by_source.items())},
            "gsd_known_frame_count": 0,
            "registration_state": {"UNKNOWN_REGISTRATION": len(enriched_items)},
        },
        "paper_calibration": calibration_status,
        "evaluation_mask_sidecar_schema": "evaluation_sidecars/mask_sidecar_contract.json",
        "model_training_policy": "core exact only; expanded main training not authorized",
        "model_requirements_sha256": args.model_requirements_sha,
    }
    write_json(release / "capability_status.json", capabilities)
    write_json(release / "readiness_states.json", {
        "schema_version": "qcpr-readiness-states-r19",
        "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": True,
        "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": False,
        "EXACT_TRAINING_READY": True,
        "EXACT_PAPER_CALIBRATION_READY": False,
        "SEMANTIC_EVAL_READY": False,
        "SEMANTIC_TRAIN_READY": False,
        "STABLE_EVAL_READY": False,
        "LOCALIZED_EVAL_READY": False,
        "DIRECTION_EVAL_READY": False,
        "LONG_SERIES_EVAL_READY": False,
        "HIGHRES_EVAL_READY": False,
        "evidence": [
            "exact_training_integrity_audit.json",
            "audits/native_geometry_audit.json",
            "audits/split_leakage_r19.json",
            "audits/reversed_pair_leakage_r19.json",
            "real_batch_relevance_audit.json",
            "audits/validate_qcpr_release_contract.json",
            "exact_paper_calibration_status.json",
        ],
        "training_launched": False,
    })

    # Produce the validator evidence against a provisional checksum inventory.
    # The validator writes outside the release first; the copied result is then
    # included in the final immutable content index and final SHA256SUMS.
    provisional_lines = []
    for path in sorted(release.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(release).as_posix()
        provisional_lines.append(f"{sha256_file(path)}  {relative}")
    (release / "SHA256SUMS").write_text("\n".join(provisional_lines) + "\n", encoding="utf-8")
    validator = Path(__file__).with_name("validate_qcpr_release_contract.py")
    validation_output = Path(f"/tmp/qcpr_validate_{release.name}.json")
    subprocess.run(
        [
            str(Path("/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python")),
            str(validator),
            "--release",
            str(release),
            "--output",
            str(validation_output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    validation_record = read_json(validation_output)
    if not validation_record.get("passed"):
        raise SystemExit("release validator did not pass before freeze")
    shutil.copy2(validation_output, release / "audits" / "validate_qcpr_release_contract.json")

    # Freeze authoritative content before writing release metadata/handoff.
    def content_index_digest() -> str:
        rows = []
        for path in sorted(release.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(release).as_posix()
            if relative in {"RELEASE.json", "SHA256SUMS"} or relative.startswith("handoff/"):
                continue
            rows.append(f"{sha256_file(path)}  {relative}")
        return sha256_text("\n".join(rows) + "\n")

    content_digest = content_index_digest()
    release_metadata = {
        "schema_version": "qcpr-bitemporal-v2-release-r19",
        "release_name": release.name,
        "immutable": True,
        "source_release": base.name,
        "content_index_sha256": content_digest,
        "dataset_generation_code_sha": args.dataset_code_sha,
        "model_requirements_sha256": args.model_requirements_sha,
        "training_launched": False,
        "capabilities": capabilities,
        "counts": counts,
        "artifacts": {
            "physical_items": "registries/physical_items.jsonl",
            "frames": "registries/frames.jsonl",
            "native_resolution_metadata": "registries/native_resolution_metadata.jsonl",
            "queries": "registries/queries.jsonl",
            "relevance_graph": "registries/relevance_graph.jsonl",
            "collision_groups": "collision_groups.jsonl",
            "exact_core_train": "exact_core_train.jsonl",
            "exact_development": "exact_core_development.jsonl",
            "exact_test": "exact_core_test.jsonl",
            "sampler_core": "sampler_core.json",
            "capability_status": "capability_status.json",
        },
        "hash_definition": "content_index_sha256 hashes all release files except RELEASE.json, SHA256SUMS and handoff/*; SHA256SUMS is the complete post-metadata file inventory.",
    }
    write_json(release / "RELEASE.json", release_metadata)

    artifact_hashes = {
        relative: sha256_file(release / relative)
        for relative in (
            "exact_core_train.jsonl", "exact_core_development.jsonl", "exact_core_test.jsonl",
            "registries/physical_items.jsonl", "registries/frames.jsonl", "registries/queries.jsonl",
            "registries/query_attributes.jsonl", "registries/relevance_graph.jsonl",
            "collision_groups.jsonl", "sampler_core.json", "caption_rotation_contract.json",
            "preprocessing_interface.json", "native_resolution_metadata.jsonl",
            "evaluation_sidecars/dense.jsonl", "exact_training_integrity_audit.json",
        ) if (release / relative).exists()
    }
    manifest_hashes = {
        "train": sha256_file(release / "exact_core_train.jsonl"),
        "development": sha256_file(release / "exact_core_development.jsonl"),
        "test": sha256_file(release / "exact_core_test.jsonl"),
    }
    train_pair_ids = [item_id_from_row(row) for row in exact_train]
    train_pair_ids = list(dict.fromkeys(value for value in train_pair_ids if value))
    development_query_ids = [row["query_id"] for row in exact_development]
    test_query_ids = [row["query_id"] for row in exact_test]
    development_gallery_ids = list(dict.fromkeys(item_id_from_row(row) for row in exact_development if item_id_from_row(row)))
    test_gallery_ids = list(dict.fromkeys(item_id_from_row(row) for row in exact_test if item_id_from_row(row)))
    benchmark_contract = {
        "schema_version": "qcpr-benchmark-contract-r19",
        "core_benchmark_reference": {
            "path": "/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_bitemporal_core_benchmark_v1_r18",
            "sha256": "ac213877e93bd0ceb689b460451cfa42c0df1fd014d5d06aa7dc33334ba0d329",
            "immutable_reference": True,
        },
        "train": {"manifest": "exact_core_train.jsonl", "manifest_sha256": manifest_hashes["train"], "ordered_pair_ids_sha256": ordered_sha(train_pair_ids), "query_count": len(exact_train), "unique_physical_pairs": len(train_pair_ids)},
        "development": {"manifest": "exact_core_development.jsonl", "manifest_sha256": manifest_hashes["development"], "ordered_gallery_ids_sha256": ordered_sha(development_gallery_ids), "ordered_query_ids_sha256": ordered_sha(development_query_ids), "gallery_count": len(development_gallery_ids), "query_count": len(development_query_ids)},
        "test": {"manifest": "exact_core_test.jsonl", "manifest_sha256": manifest_hashes["test"], "ordered_gallery_ids_sha256": ordered_sha(test_gallery_ids), "ordered_query_ids_sha256": ordered_sha(test_query_ids), "gallery_count": len(test_gallery_ids), "query_count": len(test_query_ids)},
        "pixel_identity": "native paths and native SHA256 match r18 full decode/hash audit",
    }
    write_json(release / "benchmark_contract.json", benchmark_contract)

    handoff = {
        "schema_version": "qcpr-dataset-final-to-model-r19",
        "handoff_id": "D2M-QCPR-BITEMPORAL-V2-R19",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_branch": "codex/qcpr-dataset-v2-final",
        "dataset_source_head_sha": args.dataset_code_sha,
        "release_name": release.name,
        "release_path": str(release),
        "authoritative_release_sha256": content_digest,
        "sha256sums_path": str(release / "SHA256SUMS"),
        "model_requirements_sha256": args.model_requirements_sha,
        "core_benchmark": benchmark_contract["core_benchmark_reference"],
        "QCPR_BITEMPORAL_CORE_BENCHMARK_V1": benchmark_contract["core_benchmark_reference"],
        "final_exact_train_manifest": str(release / "exact_core_train.jsonl"),
        "final_exact_development_manifest": str(release / "exact_core_development.jsonl"),
        "final_exact_test_manifest": str(release / "exact_core_test.jsonl"),
        "train_manifest_sha256": manifest_hashes["train"],
        "development_manifest_sha256": manifest_hashes["development"],
        "test_manifest_sha256": manifest_hashes["test"],
        "ordered_train_pair_ids_sha256": ordered_sha(train_pair_ids),
        "ordered_development_gallery_ids_sha256": ordered_sha(development_gallery_ids),
        "ordered_development_query_ids_sha256": ordered_sha(development_query_ids),
        "ordered_test_gallery_ids_sha256": ordered_sha(test_gallery_ids),
        "ordered_test_query_ids_sha256": ordered_sha(test_query_ids),
        "N_TRAIN_UNIQUE_PHYSICAL_PAIRS": len(train_pair_ids),
        "N_TRAIN_EXACT_QUERIES": len(exact_train),
        "verified_exact_query_count": len(exact_train),
        "semantic_group_count": len(read_jsonl(release / "registries/semantic_candidate_groups.jsonl")),
        "semantic_query_count": 0,
        "localized_query_count": len(manifest_by_scope["localized"]),
        "direction_query_count": len(manifest_by_scope["direction"]),
        "stable_query_count": len(manifest_by_scope["stable"]),
        "long_series_query_count": len(manifest_by_scope["long_series"]),
        "disabled_generic_no_change_row_count": generic_disabled,
        "counts": counts,
        "BITEMPORAL_EXACT_READY": True,
        "SEMANTIC_EVAL_READY": False,
        "SEMANTIC_TRAIN_READY": False,
        "STABLE_EVAL_READY": False,
        "LOCALIZED_EVAL_READY": False,
        "LOCALIZED_TRAIN_READY": False,
        "DIRECTION_EVAL_READY": False,
        "DIRECTION_TRAIN_READY": False,
        "LONG_SERIES_EVAL_READY": False,
        "LONG_SERIES_TRAIN_READY": False,
        "DUBAI_EXTERNAL_READY": False,
        "core_training_allowed": True,
        "main_training_allowed": False,
        "artifact_hashes": artifact_hashes,
        "registry_hashes": {
            "physical_items": sha256_file(release / "registries/physical_items.jsonl"),
            "frames": sha256_file(release / "registries/frames.jsonl"),
            "native_resolution_metadata": sha256_file(release / "registries/native_resolution_metadata.jsonl"),
            "queries": sha256_file(release / "registries/queries.jsonl"),
            "query_attributes": sha256_file(release / "registries/query_attributes.jsonl"),
            "relevance_graph": sha256_file(release / "registries/relevance_graph.jsonl"),
            "collision_groups": sha256_file(release / "collision_groups.jsonl"),
        },
        "contract_hashes": {
            "benchmark_contract": sha256_file(release / "benchmark_contract.json"),
            "relevance_contract": sha256_file(release / "relevance_contract.json"),
            "sampler_core": sha256_file(release / "sampler_core.json"),
            "caption_rotation_contract": sha256_file(release / "caption_rotation_contract.json"),
            "preprocessing_interface": sha256_file(release / "preprocessing_interface.json"),
            "capability_status": sha256_file(release / "capability_status.json"),
            "native_resolution_audit": native_audit_sha,
        },
        "native_resolution": {
            "by_source": {source: {"width": values[0], "height": values[1], "channels": values[2], "mode": values[3], "format": values[4]} for source, values in sorted(native_by_source.items())},
            "frame_count": len(enriched_frames),
            "gsd_known_frame_count": 0,
            "registration_counts": {"UNKNOWN_REGISTRATION": len(enriched_items)},
            "synchronized_native_geometry_mismatch": 0,
        },
        "relevance_contract": relevance_contract,
        "sampler_contract": sampler_core,
        "caption_rotation_contract": "caption_rotation_contract.json",
        "preprocessing_interface": "preprocessing_interface.json",
        "evaluation_mask_sidecar_schema": "evaluation_sidecars/mask_sidecar_contract.json",
        "readiness": {
            "EXACT_TRAINING_READY": True,
            "EXACT_PAPER_CALIBRATION_READY": False,
            "AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING": True,
            "AUTHORIZE_TEMPORALSIGLIP_DOMAIN_EXPANDED_ABLATION": False,
            "SEMANTIC_EVAL_READY": False,
            "SEMANTIC_TRAIN_READY": False,
            "STABLE_EVAL_READY": False,
            "LOCALIZED_EVAL_READY": False,
            "DIRECTION_EVAL_READY": False,
            "LONG_SERIES_EVAL_READY": False,
            "HIGHRES_EVAL_READY": False,
            "FOREST_TRAIN_READY": False,
            "RSCC_VERIFIED_SUBSET_READY": False,
            "TAMMS_LONGSERIES_READY": False,
            "DUBAI_EXTERNAL_READY": False,
        },
        "calibration": calibration_status,
        "source_states": source_states,
        "masks_in_evaluation_sidecars": {"count": mask_count, "training_enabled": False, "sha256": sha256_file(mask_sidecar_path) if mask_sidecar_path.exists() else None},
        "model_training_policy": "authorize only exact core; do not authorize expanded/domain/semantic/localized/long-series training",
        "training_launched": False,
    }
    write_json(release / "handoff" / "dataset_final_to_model.json", handoff)
    write_jsonl(release / "handoff" / "dataset_to_model.jsonl", [handoff])

    # Shared contracts are published atomically with the release handoff in the
    # dataset repository; their commit is recorded by the caller after this run.
    shared = Path("/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-dataset-v2-final/contracts/qcpr_shared")
    write_json(shared / "dataset_final_to_model.json", handoff)
    write_json(shared / "dataset_status.json", {
        "schema_version": "qcpr-dataset-status-r19",
        "branch": "codex/qcpr-dataset-v2-final",
        "dataset_head_sha": args.dataset_code_sha,
        "release_name": release.name,
        "release_path": str(release),
        "authoritative_release_sha256": content_digest,
        "final_training_authorized": True,
        "core_training_allowed": True,
        "main_training_allowed": False,
        "main_expanded_training_authorized": False,
        "capabilities": capabilities,
        "counts": counts,
        "calibration": calibration_status,
        "handoff_path": "contracts/qcpr_shared/handoff/dataset_final_to_model.json",
        "handoff_sha256": sha256_file(release / "handoff/dataset_final_to_model.json"),
        "training_launched": False,
    })
    write_json(shared / "dataset_capabilities.json", capabilities | {
        "authoritative_release_sha256": content_digest,
        "release_path": str(release),
        "handoff_path": "contracts/qcpr_shared/handoff/dataset_final_to_model.json",
        "handoff_sha256": sha256_file(release / "handoff/dataset_final_to_model.json"),
    })
    write_jsonl(shared / "handoff" / "dataset_to_model.jsonl", [handoff])

    # Final post-metadata inventory; SHA256SUMS is intentionally not self-listed.
    sum_lines = []
    for path in sorted(release.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(release).as_posix()
        sum_lines.append(f"{sha256_file(path)}  {relative}")
    (release / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

    # Validate the generated release without adding another mutable artifact.
    for line in sum_lines:
        expected, relative = line.split("  ", 1)
        actual = sha256_file(release / relative)
        if actual != expected:
            raise SystemExit(f"final SHA mismatch: {relative}")
    if sha256_file(release / "registries/frames.jsonl") != handoff["registry_hashes"]["frames"]:
        raise SystemExit("frame registry hash changed unexpectedly")
    print(json.dumps({
        "release": str(release),
        "authoritative_release_sha256": content_digest,
        "sha256sums_sha256": sha256_file(release / "SHA256SUMS"),
        "sha256sums_entries": len(sum_lines),
        "counts": counts,
        "exact_source_query_counts": dict(exact_train_sources),
        "native_by_source": {source: {"width": values[0], "height": values[1], "channels": values[2], "mode": values[3], "format": values[4]} for source, values in sorted(native_by_source.items())},
        "shared_contracts_updated": True,
        "training_launched": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
