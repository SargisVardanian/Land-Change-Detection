#!/usr/bin/env python3
"""Publish a truthful r20 candidate status while human review is pending.

This script intentionally refuses to manufacture an exact-training manifest.
It creates a separate candidate lineage, records immutable r19g inputs, and
emits a Model-Agent handoff with training readiness false until two independent
human reviews and adjudication are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha(value: Any) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pair_id(row: dict[str, Any]) -> str | None:
    for key in ("source_pair_id", "source_item_id", "pair_id", "item_id"):
        if row.get(key):
            return str(row[key])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--dataset-release-sha", required=True)
    parser.add_argument("--producer-git-sha", required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "handoff").mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    release = args.release_dir
    train_path = release / "exact_core_train.jsonl"
    dev_path = release / "exact_core_development.jsonl"
    test_path = release / "exact_core_test.jsonl"
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    test_rows = read_jsonl(test_path)
    r19_counts = {
        "train_queries": len(train_rows),
        "train_pairs": len({pair_id(row) for row in train_rows if pair_id(row)}),
        "development_queries": len(dev_rows),
        "development_pairs": len({pair_id(row) for row in dev_rows if pair_id(row)}),
        "test_queries": len(test_rows),
        "test_pairs": len({pair_id(row) for row in test_rows if pair_id(row)}),
    }

    audit = read_json(args.review_package / "review_package_audit.json")
    source_lineage = read_json(args.review_package / "source_lineage.json")
    decision_templates = []
    for name in ("reviewer_a_decision_template.jsonl", "reviewer_b_decision_template.jsonl", "adjudication_template.jsonl"):
        rows = read_jsonl(args.review_package / name)
        decision_templates.extend(rows)
    decisions_completed = sum(1 for row in decision_templates if row.get("final_decision") not in (None, ""))
    reviewer_decisions_completed = sum(
        1 for name in ("reviewer_a_decision_template.jsonl", "reviewer_b_decision_template.jsonl")
        for row in read_jsonl(args.review_package / name)
        if row.get("final_decision") not in (None, "")
    )
    if audit.get("dataset_release_sha") != args.dataset_release_sha:
        raise SystemExit("review package release SHA does not match requested r19g release")
    if audit.get("r19_immutable_required") is not True:
        raise SystemExit("review package does not enforce r19g immutability")
    if decisions_completed or reviewer_decisions_completed:
        raise SystemExit("this pending-report builder refuses partially populated human decisions")

    package_files = {
        name: file_sha(args.review_package / name)
        for name in (
            "review_package_audit.json",
            "source_lineage.json",
            "reviewer_a_packet.jsonl",
            "reviewer_b_packet.jsonl",
            "reviewer_a_decision_template.jsonl",
            "reviewer_b_decision_template.jsonl",
            "adjudication_template.jsonl",
        )
    }
    report = {
        "schema_version": "qcpr-r20-exact-supervision-report-v1",
        "status": "HOLD_HUMAN_REVIEW_REQUIRED",
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": args.producer_git_sha,
        "created_at": created_at,
        "dataset_release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
        "dataset_release_sha": args.dataset_release_sha,
        "r19g_immutable": True,
        "r19g_legacy_exact_eval_unchanged": True,
        "r19g_counts": r19_counts,
        "r19g_manifest_sha256": {
            "exact_core_train.jsonl": file_sha(train_path),
            "exact_core_development.jsonl": file_sha(dev_path),
            "exact_core_test.jsonl": file_sha(test_path),
        },
        "human_calibration": {
            "required_decisions": 1500,
            "required_per_stratum": 300,
            "strata": {
                "exact_discriminative": 300,
                "semantic_multi_positive": 300,
                "generic_no_change": 300,
                "stable_scene_specific": 300,
                "localized_direction": 300,
            },
            "review_package_status": audit.get("status"),
            "reviewer_decisions_completed": reviewer_decisions_completed,
            "adjudicated_rows": 0,
            "exact_identifiability_precision": None,
            "exact_identifiability_precision_ci95": None,
            "weak_caption_rate": None,
            "semantic_alternative_rate": None,
            "source_conditioned_values": None,
            "exact_review_source_counts": audit.get("exact_source_counts", {}),
            "exact_review_source_registry_sha256": audit.get("exact_source_registry_sha256"),
            "no_decisions_fabricated": True,
        },
        "r20_candidate": {
            "lineage_status": "CANDIDATE_HOLD",
            "exact_train_manifest_status": "NOT_MATERIALIZED_PENDING_HUMAN_DECISIONS",
            "exact_train_manifest_sha256": None,
            "train_query_count": None,
            "train_pair_count": None,
            "calibrated_development_manifest_status": "NOT_MATERIALIZED_PENDING_HUMAN_DECISIONS",
            "r20_matched_exact_training_ready": False,
        },
        "review_package": {
            "path": str(args.review_package),
            "source_lineage_artifact_sha256": source_lineage.get("artifact_sha256"),
            "files_sha256": package_files,
        },
        "blockers": [
            "Two independent human reviewers have not completed the 1,500-row calibration.",
            "Adjudication is not complete.",
            "Exact-identifiability precision, weak-caption rate, and semantic-alternative rate therefore have no valid human estimate.",
            "No exact-training manifest may be promoted from agent diagnostics or automated attribute heuristics.",
        ],
        "recommendation": "DO_NOT_TRAIN_YET",
        "training_authorization": {
            "main_training_allowed": False,
            "r20_matched_exact_training_ready": False,
            "model_agent_action": "DO_NOT_REQUEST_TRAINING",
        },
        "source_policy": {
            "new_sources_added": False,
            "rscc_added": False,
            "s2looking_text_added": False,
            "tamms_added": False,
            "forest_added_to_primary_training": False,
        },
        "artifact_hash_definition": "SHA256 of UTF-8 bytes of json.dumps(value excluding artifact_sha256, ensure_ascii=False, sort_keys=True, separators=(',', ':')) followed by one newline",
    }
    report["artifact_sha256"] = canonical_sha(report)
    (candidate / "qcpr_r20_exact_supervision_report.json").write_bytes(canonical_bytes(report))

    md = f"""# QCPR r20 exact-supervision report

## Status

`HOLD_HUMAN_REVIEW_REQUIRED`

The immutable r19g release remains unchanged. The 1,500-row independent review
package is ready, but both reviewer decision sheets are empty and adjudication
has not started. Agent diagnostics and automated attribute heuristics are not
used as human labels.

## Required calibration

| stratum | required | completed |
|---|---:|---:|
| exact_discriminative | 300 | 0 |
| semantic_multi_positive | 300 | 0 |
| generic_no_change | 300 | 0 |
| stable_scene_specific | 300 | 0 |
| localized_direction | 300 | 0 |
| **total** | **1500** | **0** |

Valid human estimates are therefore unavailable:

- exact-identifiability precision: `null`
- weak-caption rate: `null`
- semantic-alternative rate: `null`
- source-conditioned values: `null`

The exact-discriminative stratum is source-balanced by design: 150 unique
LEVIR physical pairs and 150 unique SECOND physical pairs. This makes the
future source-conditioned estimate identifiable, but does not create a human
estimate before review.

## r19g preservation

- `R19G_IMMUTABLE = true`
- train: {r19_counts['train_queries']} queries / {r19_counts['train_pairs']} pairs
- development: {r19_counts['development_queries']} queries / {r19_counts['development_pairs']} pairs
- test: {r19_counts['test_queries']} queries / {r19_counts['test_pairs']} pairs
- legacy development remains frozen exact evaluation

## r20 candidate

- candidate lineage: `CANDIDATE_HOLD`
- r20 exact-train manifest: **not materialized** until review/adjudication
- `R20_TRAIN_QUERY_COUNT = null`
- `R20_TRAIN_PAIR_COUNT = null`
- `R20_MATCHED_EXACT_TRAINING_READY = false`
- `DATASET_AGENT_RECOMMENDATION = DO_NOT_TRAIN_YET`

The correct next action is to have two independent human reviewers complete the
package at `{args.review_package}`, adjudicate disagreements, and rerun the
promotion builder. No model training is authorized by this report.
"""
    (candidate / "qcpr_r20_exact_supervision_report.md").write_text(md, encoding="utf-8")

    handoff = {
        "schema_version": "qcpr-dataset-final-to-model-r20-v1",
        "status": "HOLD_HUMAN_REVIEW_REQUIRED",
        "producer_agent": "DATASET_AGENT",
        "producer_git_sha": args.producer_git_sha,
        "created_at": created_at,
        "dataset_release_name": "qcpr_r20_candidate_exact_supervision_20260812",
        "parent_release_name": "qcpr_bitemporal_v2_train_20260808_final_r19g",
        "parent_dataset_release_sha": args.dataset_release_sha,
        "r19g_immutable": True,
        "r19g_legacy_exact_eval_unchanged": True,
        "R20_MATCHED_EXACT_TRAINING_READY": False,
        "main_training_allowed": False,
        "exact_train_manifest": None,
        "exact_train_manifest_sha256": None,
        "exact_train_query_count": None,
        "exact_train_pair_count": None,
        "calibrated_dev_manifest": None,
        "calibrated_dev_manifest_sha256": None,
        "human_calibration": {
            "required_decisions": 1500,
            "completed_decisions": 0,
            "review_package_audit_sha256": file_sha(args.review_package / "review_package_audit.json"),
            "reviewer_a_packet_sha256": file_sha(args.review_package / "reviewer_a_packet.jsonl"),
            "reviewer_b_packet_sha256": file_sha(args.review_package / "reviewer_b_packet.jsonl"),
            "adjudication_status": "PENDING",
            "exact_review_source_counts": audit.get("exact_source_counts", {}),
            "exact_review_source_registry_sha256": audit.get("exact_source_registry_sha256"),
        },
        "blockers": report["blockers"],
        "recommendation": "DO_NOT_TRAIN_YET",
        "artifact_hash_definition": report["artifact_hash_definition"],
    }
    handoff["artifact_sha256"] = canonical_sha(handoff)
    (candidate / "handoff/dataset_final_to_model_r20.json").write_bytes(canonical_bytes(handoff))

    (candidate / "README.md").write_text(
        "# QCPR r20 candidate exact-supervision lineage\n\n"
        "This is a HOLD lineage only. r19g is immutable. The human review package "
        "must be completed and adjudicated before an r20 exact-training manifest "
        "can be materialized. No active r20 training manifest exists in this directory.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "r19g_immutable": True,
        "reviewer_decisions_completed": 0,
        "R20_MATCHED_EXACT_TRAINING_READY": False,
        "report_artifact_sha256": report["artifact_sha256"],
        "handoff_artifact_sha256": handoff["artifact_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
