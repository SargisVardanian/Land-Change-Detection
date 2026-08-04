#!/usr/bin/env python3
"""Finalize the Stage-2 compatible architecture screen from a completed job."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--compatible-report", type=Path, required=True)
    p.add_argument("--code-sha", required=True)
    args = p.parse_args()
    audit = args.audit_root
    arch = audit / "architecture_screening"
    old_path = arch / "architecture_screening_frozen_report.json"
    old_backup = arch / "architecture_screening_frozen_report_pre_compatible.json"
    if not old_backup.exists():
        shutil.copy2(old_path, old_backup)
    old = json.loads(old_path.read_text())
    screen = json.loads(args.compatible_report.read_text())
    results = screen["architectures"]

    def screen_entry(kind: str, name: str, compatible: bool, contract: str) -> dict[str, Any]:
        item = results[kind]
        return {
            "architecture_label": name,
            "architecture_compatible_replacement": compatible,
            "score_contract": contract,
            "metrics": item["evaluation"]["metrics"],
            "training": item["training"],
            "metadata": {
                "screen_job": "205163",
                "screen_report": str(args.compatible_report),
                "screen_report_sha256": sha256(args.compatible_report),
                "checkpoint_sha256": item["training"]["checkpoint_sha256"],
                "backbone_frozen": True,
                "full_finetuning": False,
                "bounded_head_adaptation": True,
                "feature_cache_contract": screen["feature_caches"],
            },
        }

    encoders = dict(old.get("encoders", {}))
    encoders["B0_native_joint_temporal"] = screen_entry(
        "B0",
        "native joint UniverSat + residual token adapters + PAIR cross-attention",
        True,
        "indexable_global_pair_embedding",
    )
    encoders["B1_framewise_gated_difference"] = screen_entry(
        "B1",
        "framewise UniverSat + gated difference fusion + residual PAIR head",
        True,
        "indexable_global_pair_embedding",
    )
    encoders["B2_text_conditioned_temporal_fusion"] = screen_entry(
        "B2",
        "framewise UniverSat + query-conditioned temporal fusion",
        False,
        "query_conditioned_full_matrix",
    )
    b0 = results["B0"]["evaluation"]["metrics"]
    b1 = results["B1"]["evaluation"]["metrics"]
    b2 = results["B2"]["evaluation"]["metrics"]
    report = {
        "schema_version": "qcpr-stage2-architecture-screen-v3",
        "status": "SCREENING_COMPLETE_COMPATIBLE_B0_B1_B2",
        "code_sha": args.code_sha,
        "compatible_screen_job": "205163",
        "compatible_screen_report": str(args.compatible_report),
        "compatible_screen_report_sha256": sha256(args.compatible_report),
        "pre_compatible_report": str(old_backup),
        "pre_compatible_report_sha256": sha256(old_backup),
        "common_contract": screen["common_contract"],
        "backbone": screen["backbone"],
        "frozen_only_backbones": True,
        "full_finetuning": False,
        "bounded_head_adaptation": True,
        "encoders": encoders,
        "screened_architectures": [
            "B0_native_joint_temporal",
            "B1_framewise_gated_difference",
            "B2_text_conditioned_temporal_fusion",
            "B1_legacy_framewise_explicit_fusion",
            "B3_zero_shot_delta",
            "B4_zero_shot_delta",
            "SigLIP2_zero_shot_delta",
        ],
        "non_compatible_diagnostics": [
            "B2_text_conditioned_temporal_fusion is not an A0-indexable replacement because its visual representation depends on the query.",
            "B3/B4/SigLIP2 remain single-image zero-shot signed-delta diagnostics, not trained bi-temporal replacements.",
        ],
        "selection": {
            "anchor": "B1_framewise_gated_difference",
            "alternative": "B0_native_joint_temporal",
            "reason": "B1 has the strongest bounded-screen indexable MRR and R@10; B0 remains the native Stage-2 architecture alternative. B2 is excluded from A0 selection because it is query-conditioned.",
            "comparison": {
                "B0": b0,
                "B1": b1,
                "B2": b2,
            },
        },
        "scientific_caveats": [
            "The compatible screen uses 2048 train pairs, 512 development pairs, 128 fixed steps and one seed; it is an architecture screen, not a full-data claim.",
            "Metrics are exact single-positive diagnostics and do not establish semantic superiority.",
            "B1 winning this bounded screen does not authorize P2 until semantic supervision and source gates pass.",
        ],
        "job": {
            "job_id": "205163",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": "gpu03",
            "elapsed": "00:11:48",
            "peak_allocated_gib": {
                "B0": results["B0"]["training"]["peak_allocated_gib"],
                "B1": results["B1"]["training"]["peak_allocated_gib"],
                "B2": results["B2"]["training"]["peak_allocated_gib"],
            },
            "peak_reserved_gib": {
                "B0": results["B0"]["training"]["peak_reserved_gib"],
                "B1": results["B1"]["training"]["peak_reserved_gib"],
                "B2": results["B2"]["training"]["peak_reserved_gib"],
            },
        },
    }
    out_json = arch / "architecture_screening_frozen_report.json"
    out_md = arch / "architecture_screening_frozen_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    out_md.write_text(
        "\n".join(
            [
                "# Stage-2 compatible architecture screening",
                "",
                "Status: SCREENING_COMPLETE_COMPATIBLE_B0_B1_B2",
                "",
                "All UniverSat and Jina backbone parameters were frozen. The bounded screen used 128 optimizer steps, seed 20260802, 2048 train pairs, and a common 512-pair development gallery.",
                "",
                "| Architecture | MRR | R@1 | R@5 | R@10 | Mean rank | Contract |",
                "|---|---:|---:|---:|---:|---:|---|",
                f"| B0 native joint | {b0['mrr']:.6f} | {b0['r1']:.6f} | {b0['r5']:.6f} | {b0['r10']:.6f} | {b0['mean_rank']:.2f} | indexable |",
                f"| B1 gated difference | {b1['mrr']:.6f} | {b1['r1']:.6f} | {b1['r5']:.6f} | {b1['r10']:.6f} | {b1['mean_rank']:.2f} | indexable |",
                f"| B2 text-conditioned | {b2['mrr']:.6f} | {b2['r1']:.6f} | {b2['r5']:.6f} | {b2['r10']:.6f} | {b2['mean_rank']:.2f} | query-conditioned |",
                "",
                "Selected anchor: B1_framewise_gated_difference.",
                "Alternative: B0_native_joint_temporal.",
                "B2 is retained for grounding/ablation analysis and is not an A0 gallery-index replacement.",
                "",
                "This result is bounded architecture evidence, not a full-data or SOTA claim.",
            ]
        )
        + "\n"
    )
    plan_path = audit / "architecture_screening_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["code_sha"] = args.code_sha
    plan["status"] = "SCREENING_COMPLETE"
    for row in plan["architectures"]:
        if row["id"] == "B0":
            row["state"] = "SCREENED_BOUNDED_HEAD_ADAPTATION"
        elif row["id"] == "B1":
            row["state"] = "SCREENED_BOUNDED_HEAD_ADAPTATION"
        elif row["id"] == "B2":
            row["state"] = "SCREENED_BOUNDED_QUERY_CONDITIONED_ABLATION"
        else:
            row["state"] = row.get("state", "DIAGNOSTIC_ONLY")
    plan["selected_anchor"] = "B1_framewise_gated_difference"
    plan["selected_alternative"] = "B0_native_joint_temporal"
    plan["screen_report_sha256"] = sha256(out_json)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "anchor": report["selection"]["anchor"], "alternative": report["selection"]["alternative"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
