#!/usr/bin/env python3
"""Audit the retrieval-oriented Stage-2 data views without enabling training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_MASK_KEYS = {
    "mask_path",
    "pre_mask_path",
    "post_mask_path",
    "label_path",
    "dense_label_path",
    "dense_label_sidecar",
    "official_label",
    "semantic_label",
    "mask",
}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def nested_forbidden(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_MASK_KEYS:
                found.append(path)
            found.extend(nested_forbidden(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(nested_forbidden(child, f"{prefix}[{index}]"))
    return found


def view_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    forbidden = sorted({path for row in rows for path in nested_forbidden(row)})
    flags = [row.get("training_enabled") for row in rows]
    query_ids = {
        row.get("query_id") or row.get("caption_id") or row.get("text_id")
        for row in rows
        if row.get("query_id") or row.get("caption_id") or row.get("text_id")
    }
    return {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "training_enabled_rows": sum(value is True for value in flags),
        "training_disabled_rows": sum(value is False for value in flags),
        "training_flag_missing_rows": sum(value is None for value in flags),
        "unique_pairs": len({row.get("canonical_pair_id") or row.get("pair_id") for row in rows}),
        "unique_queries": len(query_ids),
        "sources": dict(sorted(Counter(str(row.get("source_dataset") or row.get("dataset_name")) for row in rows).items())),
        "forbidden_mask_fields": forbidden,
        "mask_free": not forbidden,
    }


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--forest-pilot-audit", type=Path, required=True)
    parser.add_argument("--forest-proposal-audit", type=Path, required=True)
    parser.add_argument("--tamms-pilot-audit", type=Path, required=True)
    parser.add_argument("--tamms-proposal-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = args.release / "manifests"
    views = {
        "exact_train": view_stats(manifests / "retrieval_exact_train_v2.jsonl"),
        "exact_development": view_stats(manifests / "retrieval_exact_development_v2.jsonl"),
        "exact_test": view_stats(manifests / "retrieval_exact_test_v2.jsonl"),
        "semantic_train": view_stats(manifests / "retrieval_semantic_gold_train.jsonl"),
        "semantic_development": view_stats(manifests / "retrieval_semantic_gold_development.jsonl"),
        "semantic_test": view_stats(manifests / "retrieval_semantic_gold_test.jsonl"),
        "localized_train": view_stats(manifests / "localized_instruction_train_v2.jsonl"),
        "localized_development": view_stats(manifests / "localized_instruction_development_v2.jsonl"),
        "localized_test": view_stats(manifests / "localized_instruction_test_v2.jsonl"),
        "grounding_mask_free_train": view_stats(manifests / "grounding_mask_free_train_v2.jsonl"),
        "grounding_mask_free_development": view_stats(manifests / "grounding_mask_free_development_v2.jsonl"),
        "grounding_mask_free_test": view_stats(manifests / "grounding_mask_free_test_v2.jsonl"),
        "long_series_tamms_pilot": {
            "path": str(args.tamms_pilot_audit),
            "status": read_json(args.tamms_pilot_audit, {}).get("status"),
            "sequences": read_json(args.tamms_pilot_audit, {}).get("pilot_sequence_count", 0),
            "text_rows": read_json(args.tamms_pilot_audit, {}).get("pilot_text_count", 0),
            "training_enabled": read_json(args.tamms_pilot_audit, {}).get("training_enabled", False),
        },
    }

    forest = read_json(args.forest_pilot_audit, {})
    forest_proposal = read_json(args.forest_proposal_audit, {})
    tamms = read_json(args.tamms_pilot_audit, {})
    tamms_proposal = read_json(args.tamms_proposal_audit, {})

    exact_ready = (
        views["exact_train"]["rows"] > 0
        and views["exact_development"]["rows"] > 0
        and all(views[name]["mask_free"] for name in ("exact_train", "exact_development", "exact_test"))
    )
    semantic_rows = sum(views[name]["rows"] for name in ("semantic_train", "semantic_development", "semantic_test"))
    localized_rows = sum(views[name]["rows"] for name in ("localized_train", "localized_development", "localized_test"))

    report = {
        "schema_version": "qcpr-stage2-data-view-readiness-v1",
        "release": {
            "path": str(args.release),
            "stage2_gate_status": read_json(args.release / "reports/stage2_gate_summary.json", {}).get("status"),
            "release_artifact_sha256": sha256(args.release / "hashes/stage2_release_artifacts_sha256.json"),
        },
        "target": {
            "description": "one physical pair or sequence with exact, semantic, localized, direction-sensitive, stable-scene and long-series query views",
            "views": [
                "exact_discriminative",
                "semantic_multi_positive",
                "localized",
                "direction_sensitive",
                "stable_scene",
                "long_series",
            ],
        },
        "views": views,
        "source_readiness": {
            "LEVIR-MCI": {"role": "exact_discriminative", "pairs": 8143, "status": "READY"},
            "SECOND-CC": {"role": "exact_discriminative", "pairs": 4701, "status": "READY"},
            "RSCC-EBD": {"role": "physical_temporal_candidate", "pairs": 18215, "verified_text_rows": 0, "status": "PHYSICAL_ONLY"},
            "Forest-Change": {
                "role": "forest_vegetation_candidate",
                "pairs": forest.get("pair_count", 0),
                "captions": forest.get("caption_count", 0),
                "status": forest.get("status"),
                "scene_disjoint_proposal": forest_proposal.get("status"),
                "training_enabled": forest.get("training_enabled", False),
            },
            "TAMMs": {
                "role": "long_series_candidate",
                "pilot_sequences": tamms.get("pilot_sequence_count", 0),
                "generated_text_rows": tamms.get("pilot_text_count", 0),
                "status": tamms.get("status"),
                "split_proposal": tamms_proposal.get("status"),
                "event_disjoint": tamms_proposal.get("event_disjoint", False),
                "training_enabled": tamms.get("training_enabled", False),
            },
            "TERRA-CD": {"role": "non_disaster_physical_candidate", "status": "ACCESS_REQUIRED_METADATA_ONLY"},
            "DynamicEarthNet": {"role": "long_series_physical_candidate", "status": "ACCESS_REQUIRED_METADATA_ONLY"},
            "SpaceNet-7": {"role": "long_series_physical_candidate", "status": "ACCESS_REQUIRED_METADATA_ONLY"},
        },
        "readiness": {
            "exact_view": "READY_FOR_P1" if exact_ready else "HOLD",
            "semantic_view": "HOLD_EMPTY_PRIMARY_GOLD" if semantic_rows == 0 else "REVIEW_REQUIRED",
            "localized_view": "HOLD_EMPTY" if localized_rows == 0 else "REVIEW_REQUIRED",
            "direction_sensitive_view": "NOT_SEPARATELY_VERIFIED",
            "stable_scene_view": "NOT_SEPARATELY_VERIFIED",
            "long_series_view": "HOLD_NO_OFFICIAL_EVENT_SPLIT_OR_TEXT_VERIFICATION",
            "training_authorized": False,
            "overall": "DATA_QUALITY_HOLD",
            "p2_authorized": False,
        },
        "blockers": [
            "RSCC verified text rows are zero",
            "primary semantic gold train/development/test are empty",
            "localized training queries are empty",
            "Forest official split has image-component leakage and its captions are unverified",
            "TAMMs pilot has generated unverified text and no official/event-disjoint split",
            "TERRA-CD, DynamicEarthNet and SpaceNet-7 physical assets are not locally acquired",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "stage2_data_view_readiness.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "stage2_data_view_readiness.md").write_text(
        "# QCPR Stage-2 data-view readiness\n\n"
        f"Status: {report['readiness']['overall']}\n\n"
        "The exact core is usable for P1; semantic, localized and long-series "
        "views are not training-ready. No P2 was submitted.\n\n"
        "## View status\n\n"
        "| View | Status | Rows/sequences | Training-enabled |\n"
        "|---|---|---:|---:|\n"
        f"| Exact | {report['readiness']['exact_view']} | {views['exact_train']['rows']} train / {views['exact_development']['rows']} dev | {views['exact_train']['training_enabled_rows']} / {views['exact_development']['training_enabled_rows']} |\n"
        f"| Semantic gold | {report['readiness']['semantic_view']} | {semantic_rows} | {sum(views[name]['training_enabled_rows'] for name in ('semantic_train', 'semantic_development', 'semantic_test'))} |\n"
        f"| Localized | {report['readiness']['localized_view']} | {localized_rows} | {sum(views[name]['training_enabled_rows'] for name in ('localized_train', 'localized_development', 'localized_test'))} |\n"
        f"| TAMMs long-series pilot | {report['readiness']['long_series_view']} | {tamms.get('pilot_sequence_count', 0)} sequences / {tamms.get('pilot_text_count', 0)} text rows | {tamms.get('training_enabled', False)} |\n\n"
        "## Blockers\n\n"
        + "".join(f"- {item}\n" for item in report["blockers"])
        + "\n"
        "Forest and TAMMs proposals are deliberately not releases; they preserve "
        "the evidence needed for the next audit step without silently enabling training.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": report["readiness"]["overall"], "semantic_rows": semantic_rows, "localized_rows": localized_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
