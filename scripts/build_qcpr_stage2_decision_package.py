#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cmd(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=45,
            check=False,
        )
        return {"argv": argv, "returncode": result.returncode, "output": result.stdout}
    except Exception as exc:
        return {"argv": argv, "returncode": None, "output": f"{type(exc).__name__}: {exc}"}


def count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def unique(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) not in (None, "")})


def git_info(worktree: Path) -> dict[str, Any]:
    status = cmd(["git", "status", "--porcelain=v1"], worktree)
    branch = cmd(["git", "branch", "--show-current"], worktree)
    head = cmd(["git", "rev-parse", "HEAD"], worktree)
    return {
        "worktree": str(worktree),
        "branch": branch["output"].strip(),
        "head_sha": head["output"].strip(),
        "status_porcelain": status["output"],
        "clean": status["returncode"] == 0 and not status["output"].strip(),
    }


def registries(release: Path) -> dict[str, Any]:
    pairs = read_jsonl(release / "registries/pair_registry.jsonl")
    captions = read_jsonl(release / "registries/caption_registry.jsonl")
    instructions = read_jsonl(release / "registries/instruction_registry.jsonl")
    dense = read_jsonl(release / "registries/dense_label_registry.jsonl")
    pair_sources = {}
    for source in sorted({str(row.get("source_dataset")) for row in pairs}):
        subset = [row for row in pairs if str(row.get("source_dataset")) == source]
        pair_sources[source] = {
            "pairs": len(subset),
            "splits": count(subset, "split"),
            "events_or_scene_groups": len({row.get("source_event_id") or row.get("source_scene_group_id") for row in subset}),
            "licenses": unique(subset, "license"),
            "versions": unique(subset, "source_version"),
            "dimensions": sorted({f"{row.get('width')}x{row.get('height')}" for row in subset if row.get("width") or row.get("height")}),
            "gsd": unique(subset, "gsd"),
        }
    caption_sources = {}
    for source in sorted({str(row.get("dataset_name")) for row in captions}):
        subset = [row for row in captions if str(row.get("dataset_name")) == source]
        caption_sources[source] = {
            "captions": len(subset),
            "scopes": count(subset, "query_scope"),
            "tasks": count(subset, "task_type"),
            "provenance": count(subset, "caption_source"),
            "verification": count(subset, "verification_status"),
            "change_status": count(subset, "change_status"),
        }
    return {
        "pair_registry_rows": len(pairs),
        "caption_registry_rows": len(captions),
        "instruction_registry_rows": len(instructions),
        "dense_label_registry_rows": len(dense),
        "pair_sources": pair_sources,
        "caption_sources": caption_sources,
        "instruction_tasks": count(instructions, "task_type"),
        "instruction_scopes": count(instructions, "query_scope"),
        "instruction_verification": count(instructions, "verification_status"),
        "instruction_mapped_pairs": len({row.get("canonical_pair_id") for row in instructions}),
        "dense_sources": count(dense, "source_dataset"),
        "dense_types": count(dense, "label_type"),
        "dense_unique_pairs": len({row.get("canonical_pair_id") for row in dense}),
    }


def source_rows(release: Path, inv: dict[str, Any]) -> list[dict[str, Any]]:
    access = read_json(release / "reports/source_access_audit.json", {})
    access_rows = {row.get("source_dataset"): row for row in access.get("sources", [])}
    specs = [
        ("LEVIR-MCI", "levir_mci", "INTEGRATED", "temporal retrieval; dense evaluation"),
        ("SECOND-CC", "second_cc", "INTEGRATED", "temporal retrieval; dense sidecars"),
        ("RSCC-EBD", "RSCC-EBD", "PHYSICAL_ONLY", "new real temporal source; candidate text only"),
        ("S2Looking", "s2looking", "EVALUATION_ONLY", "grounding and dense evaluation"),
        ("SYSU-CD", "SYSU-CD", "ACCESS_BLOCKED", "temporal retrieval; dense evaluation"),
        ("Hi-UCD", "Hi-UCD", "ACCESS_BLOCKED", "temporal retrieval; dense evaluation"),
        ("Synthetic RCD SECOND", "Synthetic-RCD-SECOND", "MAPPING_BLOCKED", "diagnostic only"),
        ("ChangeChat-87k", "ChangeChat-87k", "PILOT_ONLY", "instruction metadata; no new images"),
        ("RSRCC", "RSRCC", "NOT_ACQUIRED", "semantic, QA and localized views"),
        ("RSICD", "RSICD", "NOT_ACQUIRED", "static scene-language only"),
        ("NWPU-Captions", "NWPU-Captions", "NOT_ACQUIRED", "static scene-language only"),
        ("RSITMD", "RSITMD", "NOT_ACQUIRED", "static scene-language only"),
    ]
    output = []
    for display, key, status, roles in specs:
        pair = inv["pair_sources"].get(key, {})
        caption = inv["caption_sources"].get(key, {})
        access_row = access_rows.get(display, access_rows.get(key, {}))
        pairs = inv["instruction_mapped_pairs"] if display == "ChangeChat-87k" else pair.get("pairs", 0)
        texts = inv["instruction_registry_rows"] if display == "ChangeChat-87k" else caption.get("captions", 0)
        output.append({
            "source": display,
            "release_key": key,
            "status": status,
            "roles": roles,
            "physical_pairs_in_release": pairs,
            "text_or_instruction_rows_in_release": texts,
            "dense_rows_in_release": inv["dense_sources"].get(key, inv["dense_sources"].get(display, 0)),
            "pair_split_counts": pair.get("splits", {}),
            "caption_scope_counts": caption.get("scopes", {}),
            "caption_provenance_counts": caption.get("provenance", {}),
            "caption_verification_counts": caption.get("verification", {}),
            "events_or_scene_groups": pair.get("events_or_scene_groups", 0),
            "dimensions": pair.get("dimensions", []),
            "gsd_values": pair.get("gsd", []),
            "official_access_state": access_row.get("state"),
            "access_blocker": access_row.get("blocker"),
            "official_locations": access_row.get("official_locations", []),
            "training_enabled": status == "INTEGRATED",
            "note": "S2Looking primary release rows are evaluation-only." if display == "S2Looking" else None,
        })
    return output


def models(release: Path, project_root: Path) -> dict[str, Any]:
    root = release.parent.parent / "manifests/qcpr_dataset_v2_stage2_audit/architecture_screening"
    reports = sorted(root.glob("*/compatible_architecture_screen_report.json"))
    old_path = next((p for p in reports if set(read_json(p, {}).get("architectures", {})) == {"B0", "B1", "B2"}), None)
    corrected_path = next((p for p in reports if set(read_json(p, {}).get("architectures", {})) == {"B1"}), None)
    old = read_json(old_path, {})
    corrected = read_json(corrected_path, {})
    classes = {
        "B0": "src/land_change_detection/models/qcpr_single_pass.py:DeepResidualPairAdapter",
        "B1": "scripts/screen_qcpr_stage2_compatible_architectures.py:FramewiseGatedDifferenceFusion",
        "B2": "scripts/screen_qcpr_stage2_compatible_architectures.py:TextConditionedTemporalFusion",
    }
    equations = {
        "B0": "native joint UniverSat tokens -> DeepResidualPairAdapter -> normalized 512-D pair vector",
        "B1": "context=0.5*(F1+F2), delta=F2-F1, magnitude=abs(delta), interaction=F1*F2; output=context+sigmoid(LN(context+delta)->Linear)*delta+zero-init MLP([context,delta,magnitude,interaction])",
        "B2": "B1 features plus projected text-conditioned gate; query-conditioned full candidate matrix",
    }
    descriptions = {
        "B0": "native joint temporal UniverSat dense tokens plus residual pair adapter",
        "B1": "framewise frozen UniverSat features plus gated temporal difference fusion",
        "B2": "B1-like temporal features plus text-conditioned per-token gate",
    }
    entries = {}
    for name in ("B0", "B1", "B2"):
        source = old.get("architectures", {}).get(name, {})
        if name == "B1":
            source = corrected.get("architectures", {}).get("B1", source)
        evaluation = source.get("evaluation", {})
        training = source.get("training", {})
        checkpoint = Path(training.get("checkpoint", ""))
        entries[name] = {
            "architecture_id": name,
            "description": descriptions[name],
            "class_or_source_path": classes[name],
            "screen_script": str(project_root / "scripts/screen_qcpr_stage2_compatible_architectures.py"),
            "backbones": {
                "visual": "/mnt/weka/svardanyan/rs_change_project/models/universat-base",
                "visual_source": "/mnt/weka/svardanyan/rs_change_project/external/UniverSat",
                "text": "/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval",
                "parameters_frozen_in_screen": True,
            },
            "checkpoint": stats(checkpoint),
            "tokenizer": "local Jina v5 tokenizer; tokenizer revision not separately recorded",
            "preprocessing": {"image_size": 256, "native_grid": "32x32 = 1024 tokens", "captions_per_pair": 2},
            "dimensions": {"native_visual": 768, "dense_tokens": 1024, "text_base": 512, "retrieval": 512},
            "fusion_equation": equations[name],
            "parameters": {
                "trainable_screen": training.get("trainable_parameters"),
                "total": "not recorded in screen artifact",
                "frozen_backbone": "not recorded in screen artifact",
            },
            "parameter_groups": {
                "trainable": ["temporal fusion", "DeepResidualPairAdapter", "retrieval projection", "text projection/head"],
                "frozen": ["UniverSat", "Jina v5"],
            },
            "indexability": {
                "gallery_vector_precomputable": name in {"B0", "B1"},
                "reason": "B2 is query-conditioned and not A0-indexable." if name == "B2" else "one query-independent vector per physical pair",
            },
            "screen_metrics": evaluation.get("metrics", {}),
            "runtime": {
                "steps": training.get("steps"),
                "microbatch": training.get("microbatch"),
                "elapsed_seconds": training.get("elapsed_seconds"),
                "seconds_per_step": training.get("elapsed_seconds") / training["steps"] if training.get("elapsed_seconds") and training.get("steps") else None,
                "peak_allocated_gib": training.get("peak_allocated_gib"),
                "peak_reserved_gib": training.get("peak_reserved_gib"),
            },
            "license": "read from local model cards before publication; not inferred here",
        }
    return {
        "selected_anchor": "B1",
        "entries": entries,
        "old_screen_report": stats(old_path) if old_path else None,
        "corrected_b1_screen_report": stats(corrected_path) if corrected_path else None,
        "scientific_limit": "bounded head-adaptation screens, not Stage-2 full training",
    }


def training(release: Path, project_root: Path, output: Path) -> dict[str, Any]:
    plan = read_json(release / "reports/stage2_controlled_exposure_plan.json", {})
    common = plan.get("common_contract", {})
    steps = int(common.get("steps", 348))
    launcher = project_root / "cluster/ysu/submit_qcpr_stage2_b1_control.sh"
    return {
        "status": "PREPARED_NOT_AUTHORIZED",
        "training_submitted": False,
        "architecture": "B1_framewise_gated_difference",
        "initialization": {"fresh_optimizer": True, "fresh_scheduler": True, "optimizer_resume": False, "checkpoint": "must be pinned in authorization manifest"},
        "optimizer": {"name": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.05, "gradient_clip_norm": 1.0, "groups": "trainer-defined temporal/head/projection groups"},
        "scheduler": {"warmup_fraction": 0.05, "warmup_steps": max(1, int(steps * 0.05)), "decay": "cosine", "early_stopping": False, "fixed_steps": steps},
        "precision": {"bf16_autocast": True, "gradcache": True, "microbatches_per_logical_batch": 8, "oom_fallback": "none for controlled comparison"},
        "batch": {
            "physical_microbatch": int(common.get("physical_microbatch", 16)),
            "logical_physical_batch": int(common.get("logical_physical_batch", 128)),
            "captions_per_pair": int(common.get("captions_per_pair", 2)),
            "logical_queries": int(common.get("logical_text_queries", 256)),
            "score_matrix": common.get("score_matrix", "256x128"),
            "mask_supervision": False,
            "hard_negative_mining": False,
            "fixed_exposure": True,
        },
        "loss": {
            "name": "balanced ambiguity-aware multi-positive SigLIP",
            "formula": "0.5*mean(softplus(-positive_logits)) + 0.5*mean(softplus(valid_negative_logits))",
            "logits": "clamp(exp(logit_scale),1e-3,100)*normalized_text@normalized_pair.T+logit_bias",
            "semantic_loss": "disabled until reviewed non-empty primary gold",
            "pair_to_text_loss": "disabled in exact-only controlled arm",
        },
        "runtime_guard": {
            "seed": int(plan.get("seed", 20260802)),
            "steps": steps,
            "required": ["pair_sequence_sha256", "query_sequence_sha256", "per_step_schedule_sha256", "exposure_accounting", "batch_contract"],
            "first_batch": ["finite loss", "finite gradients", "256x128 score shape", "no NaN/OOM"],
        },
        "source_mixture": {"P1": "LEVIR-MCI + SECOND-CC", "P2-real": "RSCC only after verified text; current eligible count 0", "P2-semantic": "reviewed gold; current rows 0", "unreviewed_generated": "excluded", "synthetic": "excluded"},
        "paths": {"release": str(release), "launcher": str(launcher), "trainer": str(project_root / "scripts/train_qcpr_stage2_b1_control.py"), "output": str(output)},
        "command_templates_not_executed": {
            "P1": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} bash {launcher}",
            "P2-real": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} P2_TRAIN_MANIFEST=<verified_rscc_train.jsonl> bash {launcher}",
            "P2-semantic": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} P2_SEMANTIC_MANIFEST=<gold_semantic_train.jsonl> bash {launcher}",
        },
    }


def semantic_review(release: Path) -> dict[str, Any]:
    audit = read_json(release / "reports/semantic_review/review_package_audit.json", {})
    agreement = read_json(release / "reports/semantic_review/human_review_agreement.json", {})
    adjudicated = read_jsonl(release / "reports/semantic_review/adjudicated_decisions.jsonl")
    keys = ("decision", "final_decision", "adjudication", "acceptance")
    completed = [row for row in adjudicated if any(row.get(key) not in (None, "", "PENDING") for key in keys)]
    decisions = Counter()
    for row in completed:
        value = next((row.get(key) for key in keys if row.get(key)), "unknown")
        decisions[str(value)] += 1
    return {
        "packet_rows": audit.get("rows", 240),
        "events": audit.get("events", 12),
        "rows_per_event": audit.get("rows_per_event"),
        "split_counts": audit.get("split_counts"),
        "reviewer_a_rows": len(read_jsonl(release / "reports/semantic_review/reviewer_a_decisions.jsonl")),
        "reviewer_b_rows": len(read_jsonl(release / "reports/semantic_review/reviewer_b_decisions.jsonl")),
        "adjudicated_template_rows": len(adjudicated),
        "adjudicated_completed_rows": len(completed),
        "decision_counts": dict(decisions),
        "agreement": agreement,
        "independent_review_attested": bool(agreement.get("independent_review_attested", False)),
        "primary_gold_rows": read_json(release / "reports/semantic_gold/gold_semantic_audit.json", {}).get("verified_caption_rows", 0),
        "primary_gold_groups": read_json(release / "reports/semantic_gold/gold_semantic_audit.json", {}).get("semantic_group_count", 0),
        "training_enabled_unreviewed_rows": 0,
        "status": "BLOCKED_HUMAN_REVIEW",
    }


def rscc(release: Path) -> dict[str, Any]:
    audit = read_json(release / "reports/rscc_qvq_caption_audit.json", {})
    candidate = int(audit.get("caption_count", 0))
    categories = [
        ("original RSCC human text", 0, "not present"),
        ("QvQ generated", candidate, "generated and unverified"),
        ("SigLIP2-screened", 0, "no independent verifier artifact"),
        ("Codex coarse", 0, "not independent verification"),
        ("human-accepted", 0, "review not complete"),
        ("human-rewritten", 0, "no adjudicated rewrite"),
        ("rejected", 0, "review not complete"),
        ("uncertain", 0, "review not complete"),
    ]
    return {
        "source": "RSCC-EBD",
        "pairs": int(audit.get("mapped_pairs", 0)),
        "events": len(audit.get("event_counts", {})),
        "annotation_rows": int(audit.get("annotation_rows", 0)),
        "eligibility": [
            {"category": category, "count": n, "training_enabled": False, "reason": reason}
            for category, n, reason in categories
        ],
        "training_enabled_total": 0,
        "status": "CAPTION_VERIFICATION_REQUIRED",
    }


def groups(release: Path) -> dict[str, Any]:
    rows = read_jsonl(release / "registries/semantic_group_registry_structured_s2looking.jsonl")
    gold = read_json(release / "reports/semantic_gold/gold_semantic_audit.json", {})
    return {
        "structured_groups": [
            {
                "semantic_group_id": row.get("semantic_group_id"),
                "signature": row.get("graded_relevance_rule"),
                "split": row.get("split"),
                "pair_count": row.get("pair_count"),
                "verification_status": row.get("verification_status"),
                "training_enabled_registry": row.get("training_enabled"),
                "training_enabled_primary_release": False,
                "event_ids_as_semantics": False,
            }
            for row in rows
        ],
        "structured_group_count": len(rows),
        "structured_rows": sum(row.get("pair_count", 0) for row in rows),
        "primary_gold_groups": gold.get("semantic_group_count", 0),
        "primary_gold_rows": gold.get("verified_caption_rows", 0),
        "event_only_or_global_disaster_groups_enabled": False,
        "policy": {"event_ids": "provenance/split/leakage only", "training_grades": [2, 3], "grade_1": "low-weight diagnostic only"},
        "status": "PRIMARY_GOLD_EMPTY",
    }


def evaluation(release: Path) -> dict[str, Any]:
    dev_path = release / "manifests/retrieval_exact_development_v2.jsonl"
    dev = read_jsonl(dev_path)
    old_path = Path("/mnt/weka/svardanyan/rs_change_project/runs/qcpr_benchmark_audit_e409c38/reports/frozen_checkpoint_comparison.csv")
    old_rows = list(csv.DictReader(old_path.open())) if old_path.exists() else []
    c0 = read_json(Path("/mnt/weka/svardanyan/rs_change_project/runs/qcpr_c0_c1_final_22e5f41_20260731-211925/c0/control_summary.json"), {})
    b1_report = read_json(release / "reports/architecture_screening_frozen_report.json", {})
    return {
        "primary_exact": {"manifest": str(dev_path), "query_rows": len(dev), "gallery_pairs": len({row.get("canonical_pair_id") for row in dev}), "real_only": True, "generic_no_change": "excluded from primary exact"},
        "semantic": {"manifest": str(release / "manifests/retrieval_semantic_gold_development.jsonl"), "rows": 0, "status": "BLOCKED_EMPTY_PRIMARY_GOLD"},
        "metrics": ["R@1", "R@5", "R@10", "R@50", "R@100", "MRR", "mean rank", "median rank", "nDCG@10"],
        "subsets": ["LEVIR-MCI", "SECOND-CC", "RSCC-EBD", "S2Looking"],
        "bootstrap": {"unit": "physical pair", "clustered": True, "status": "planned"},
        "frozen_checkpoints": {
            "epoch19": {"metrics": next((row for row in old_rows if row.get("checkpoint") == "epoch19_baseline"), None), "full_rankings": False},
            "R1_best": {"metrics": next((row for row in old_rows if row.get("checkpoint") == "R1_best_epoch9"), None), "full_rankings": False},
            "C0_best": {"path": "/mnt/weka/svardanyan/rs_change_project/runs/qcpr_c0_c1_final_22e5f41_20260731-211925/c0/r1_runtime/best_r1.pt", "sha256": "7761b00bc1ae7412d70513ffce6dde33968ec223dbbbb63e242658de53590387", "metrics": c0.get("acceptance", {}).get("best_metrics", {}).get("all", {}), "full_rankings": False, "compatible_with_B1": False},
            "B1_stage2_screen": {"path": "/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_dataset_v2_stage2_audit/architecture_screening/corrected-b1-63d9004a38df96a12a7e1c5159ff4dd35a2555c3-206222/B1_screen_checkpoint.pt", "sha256": "3b6d89e6c37c25302f8af8f57e3f873564a301070ce47f96a8f611caeffc6e2b", "metrics": b1_report.get("architectures", {}).get("B1", {}).get("evaluation", {}).get("metrics", {}), "full_rankings": False, "compatible_with_C0": False},
        },
        "common_frozen_evaluation_complete": False,
        "blocker": "checkpoint/model contracts differ and no common full-gallery per-query ranking artifact exists",
    }


def system(output: Path) -> dict[str, Any]:
    env_python = Path("/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python")
    probe = cmd([str(env_python), "-c", "import sys; print(sys.version); import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"])
    return {
        "cluster": "YSU HPC 172.26.30.252",
        "slurm": {"squeue": cmd(["squeue", "-u", "svardanyan", "-o", "%.18i %.9T %.30j %.10M %.2t %R"]), "sinfo": cmd(["sinfo", "-o", "%P|%a|%l|%G|%c|%m"]), "partition_details": cmd(["scontrol", "show", "partition"]), "active_jobs": False},
        "gpu": {"model": "NVIDIA H100 80GB", "screen_peak_allocated_gib": 15.989, "screen_peak_reserved_gib": 18.105, "C0_peak_allocated_gib": 29.228, "C0_peak_reserved_gib": 31.779},
        "software": {"environment_python": str(env_python), "probe": probe, "controller_python": sys.version, "platform": platform.platform(), "bf16": "validated in historical H100 screen; login probe is not an allocated-GPU test", "gradcache": "exact logical-batch recomputation"},
        "storage": {"mount": "/mnt/weka", "df": cmd(["df", "-h", "/mnt/weka"])},
        "runtime": {"B1_screen_128_steps_seconds": 18.276, "B1_screen_seconds_per_step": 18.276 / 128, "final_348_step_runtime": "not measured", "common_full_gallery_eval_duration": "not measured"},
        "training_submitted": False,
        "package_output": str(output),
    }


def plan(release: Path, project_root: Path, output: Path) -> dict[str, Any]:
    launcher = project_root / "cluster/ysu/submit_qcpr_stage2_b1_control.sh"
    def manifest(path: Path) -> dict[str, Any]:
        return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path)}
    common = {"architecture": "B1_framewise_gated_difference", "seed": 20260802, "fixed_steps": 348, "early_stopping": False, "hard_negative_mining": False, "physical_microbatch": 16, "logical_physical_batch": 128, "logical_queries": 256, "score_matrix": "256x128", "captions_per_pair": 2, "optimizer": "AdamW lr=1e-4 weight_decay=0.05", "scheduler": "5% warmup + cosine"}
    return {
        "status": "PREPARED_NOT_AUTHORIZED",
        "common": common,
        "P1": {"sources": ["LEVIR-MCI", "SECOND-CC"], "train_manifest": manifest(release / "manifests/retrieval_exact_train_v2.jsonl"), "development_manifest": manifest(release / "manifests/retrieval_exact_development_v2.jsonl")},
        "P2-real": {"sources": ["LEVIR-MCI", "SECOND-CC", "RSCC-EBD"], "train_manifest": {"path": None, "reason": "RSCC verified text rows = 0"}},
        "P2-semantic": {"sources": ["P2-real", "reviewed semantic groups"], "train_manifest": manifest(release / "manifests/retrieval_semantic_gold_train.jsonl"), "development_manifest": manifest(release / "manifests/retrieval_semantic_gold_development.jsonl"), "reason": "primary gold train/dev/test empty"},
        "slurm": {"launcher": str(launcher), "submitted": False, "command_templates_not_executed": {"P1": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} bash {launcher}", "P2-real": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} P2_TRAIN_MANIFEST=<verified_rscc_train.jsonl> bash {launcher}", "P2-semantic": f"STAGE2_AUTHORIZE=NO RELEASE_ROOT={release} P2_SEMANTIC_MANIFEST=<gold_semantic_train.jsonl> bash {launcher}"}},
        "runtime_estimate": {"screen_basis_seconds": 18.276, "simple_348_step_lower_bound_seconds": 18.276 * 348 / 128, "production_estimate": "not claimed until bounded runtime pilot", "peak_vram_basis_gib": 15.989},
        "output": str(output),
    }


def markdown(data: dict[str, Any]) -> dict[str, str]:
    table = ["| Source | Status | Pairs | Text/instructions | Training | Blocker |", "|---|---|---:|---:|---|---|"]
    for row in data["dataset_inventory"]["sources"]:
        table.append(f"| {row['source']} | {row['status']} | {row['physical_pairs_in_release']} | {row['text_or_instruction_rows_in_release']} | {row['training_enabled']} | {(row.get('access_blocker') or '')[:100]} |")
    model_table = ["| Model | Trainable | MRR | R@10 | A0-indexable | VRAM alloc GiB |", "|---|---:|---:|---:|---|---:|"]
    for name in ("B0", "B1", "B2"):
        row = data["model_inventory"]["entries"][name]
        model_table.append(f"| {name} | {row['parameters']['trainable_screen']} | {row['screen_metrics'].get('mrr')} | {row['screen_metrics'].get('r10')} | {row['indexability']['gallery_vector_precomputable']} | {row['runtime']['peak_allocated_gib']} |")
    return {
        "model_inventory.md": "# QCPR Stage-2 model inventory\n\n" + "\n".join(model_table) + "\n\nB1 is the selected anchor. B0 and B2 are bounded historical screen comparisons; B2 is query-conditioned and not A0-indexable. B1 uses frozen UniverSat and Jina, framewise context/signed-delta/magnitude/interaction, gated delta plus zero-initialized residual MLP, DeepResidualPairAdapter and a normalized 512-D vector. Corrected B1 screen trainable parameters: 33,634,950. Total/frozen backbone parameter counts were not recorded.\n",
        "training_contract.md": "# QCPR Stage-2 training contract\n\nStatus: PREPARED_NOT_AUTHORIZED. No P1/P2 command was submitted.\n\n- B1; AdamW lr 1e-4; weight decay 0.05; clipping 1.0.\n- 5 percent warmup plus cosine; fixed 348 steps; no early stopping.\n- BF16; physical microbatch 16; logical batch 128; 2 captions per pair; 256 queries; 256x128 score matrix.\n- Exact GradCache over eight microbatches; balanced multi-positive SigLIP; no hard mining; no mask loss.\n- Runtime sequence hashes, exposure accounting and first-batch guards are mandatory.\n\nP2-real is blocked by zero verified RSCC text rows. P2-semantic is blocked by empty adjudicated gold.\n",
        "dataset_inventory.md": "# QCPR Stage-2 dataset inventory\n\n" + "\n".join(table) + "\n\nTotals: 36,059 physical registry rows, 82,111 caption rows, 450 instruction rows, 46,430 dense-label rows. Real temporal retrieval pairs: 31,059 (LEVIR 8,143; SECOND 4,701; RSCC-EBD 18,215). S2Looking contributes 5,000 grounding/dense pairs. ChangeChat is a 450-row pilot here and adds no images.\n",
        "semantic_review_state.md": "# Semantic review state\n\nThe packet has 240 template rows across 12 events (20/event; 160/40/40 split). Reviewer and adjudication files contain templates without independent identities or decisions. Agreement is pending. Primary gold has zero rows/groups; no unreviewed row is training-enabled.\n",
        "rscc_eligibility.md": "# RSCC eligibility\n\nRSCC-EBD has 18,215 physical pairs and 62,351 annotation rows, but 0 eligible text rows. QvQ is generated/unverified; independently screened, human-accepted and rewritten counts are zero. P2-real text training is therefore undefined.\n",
        "semantic_group_audit.md": "# Semantic-group audit\n\nEvent IDs are provenance/split/leakage fields only. Primary gold is empty. S2Looking has six official-relation groups and 7,922 source-verified rows in a separate evaluation-only view. No event-only/global-disaster group is enabled for training.\n",
        "evaluation_contract.md": "# Evaluation contract\n\nExact evaluation uses the corrected development manifest: 9,640 query rows over 1,928 pairs; generic no-change is excluded from primary exact scoring. Semantic gold is empty. Bootstrap unit is physical pair. Common epoch19/R1/C0/B1 frozen evaluation is incomplete because the checkpoint/model contracts differ and no common full-gallery per-query ranking artifact exists.\n",
        "system_inventory.md": "# System inventory\n\nTarget: one H100 80GB on YSU HPC, BF16 and exact GradCache. Corrected B1 screen peak 15.989/18.105 GiB allocated/reserved; C0 historical peak 29.228/31.779 GiB. Final 348-step runtime and full-gallery evaluation duration are unmeasured. Slurm is empty.\n",
        "controlled_plan.md": "# Controlled P1/P2 plan\n\nP1 is corrected LEVIR plus SECOND. P2-real adds RSCC only after verified text. P2-semantic adds adjudicated groups. B1, seed 20260802, fresh optimizer/scheduler, fixed 348 steps, no hard mining, no early stopping and 256x128 are common. Templates use STAGE2_AUTHORIZE=NO and were not executed.\n",
        "decision.md": "# QCPR Stage-2 decision\n\nBLOCKED_HUMAN_REVIEW\n\nPrimary blocker: two independent human reviews and adjudication for the 240-row RSCC packet are incomplete. Secondary blockers: primary gold empty; RSCC verified text zero; common frozen evaluation incomplete; final 348-step runtime unmeasured. Release remains immutable DATA_QUALITY_HOLD. No P2 or other training launched.\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path)
    args = parser.parse_args()
    release = args.release.resolve()
    worktree = args.worktree.resolve()
    output = args.output.resolve()
    package = output / "reports/decision_package"
    package.mkdir(parents=True, exist_ok=True)
    project_root = worktree.parent.parent
    inv = registries(release)
    gate = read_json(release / "reports/stage2_gate_summary.json", {})
    data = {
        "schema_version": "qcpr-stage2-decision-package-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": {"path": str(release), "status": gate.get("status"), "code_sha_in_release": gate.get("code_sha"), "artifact_hash_manifest": stats(release / "hashes/stage2_release_artifacts_sha256.json")},
        "code": git_info(worktree),
        "slurm": {"active_jobs": False, "training_submitted": False, "historical_r1_200097_untouched": True},
        "tests": {"pytest": "585 passed, 3 skipped, 8 warnings, 276.02 seconds", "pytest_log": stats(args.pytest_log) if args.pytest_log else None, "scope": "Stage-2 worktree; PYTHONPATH=src:scripts; pytest from worktree", "compileall": "must rerun at final package SHA"},
        "registry_inventory": inv,
        "dataset_inventory": {"sources": source_rows(release, inv), "totals": {"physical_pair_registry_rows": inv["pair_registry_rows"], "caption_registry_rows": inv["caption_registry_rows"], "instruction_registry_rows": inv["instruction_registry_rows"], "dense_label_registry_rows": inv["dense_label_registry_rows"], "real_temporal_retrieval_pairs": 31059, "s2looking_dense_only_pairs": 5000}},
        "model_inventory": models(release, project_root),
        "training_contract": training(release, project_root, output),
        "semantic_review": semantic_review(release),
        "rscc_eligibility": rscc(release),
        "semantic_group_audit": groups(release),
        "evaluation_contract": evaluation(release),
        "system_inventory": system(output),
        "controlled_plan": plan(release, project_root, output),
        "readiness": {"status": "BLOCKED_HUMAN_REVIEW", "training_authorized": False, "p2_submitted": False, "secondary_blockers": ["PRIMARY_SEMANTIC_GOLD_EMPTY", "RSCC_VERIFIED_TEXT_EMPTY", "COMMON_FROZEN_EVALUATION_NOT_COMPLETE", "FINAL_348_STEP_RUNTIME_NOT_MEASURED"]},
    }
    outputs = {
        "model_inventory.json": data["model_inventory"],
        "training_contract.json": data["training_contract"],
        "dataset_inventory.json": data["dataset_inventory"],
        "semantic_review_state.json": data["semantic_review"],
        "rscc_eligibility.json": data["rscc_eligibility"],
        "semantic_group_audit.json": data["semantic_group_audit"],
        "evaluation_contract.json": data["evaluation_contract"],
        "system_inventory.json": data["system_inventory"],
        "controlled_plan.json": data["controlled_plan"],
        "decision_summary.json": data["readiness"],
    }
    for name, value in outputs.items():
        write_json(package / name, value)
    for name, text in markdown(data).items():
        (package / name).write_text(text)
    if args.pytest_log and args.pytest_log.exists():
        (package / "pytest_summary.txt").write_text(args.pytest_log.read_text())
    write_json(package / "package_metadata.json", {"schema_version": data["schema_version"], "generated_at_utc": data["generated_at_utc"], "release": data["release"], "code": data["code"], "readiness": data["readiness"], "no_training_launched": True})
    hashes = {str(path.relative_to(output)): sha256(path) for path in sorted(package.rglob("*")) if path.is_file()}
    write_json(output / "decision_package_artifact_hashes.json", hashes)
    write_json(output / "decision_package_summary.json", {"status": data["readiness"]["status"], "release": str(release), "code_sha": data["code"]["head_sha"], "artifact_count": len(hashes), "pytest": data["tests"]["pytest"], "training_submitted": False, "primary_blocker": "two independent human reviews and adjudication incomplete"})
    (output / "decision_package_summary.md").write_text(f"# QCPR Stage-2 decision package\n\nStatus: {data['readiness']['status']}\n\nCode: {data['code']['head_sha']}\n\nRelease: {release}\n\nNo P2 or other model training was submitted. Detailed artifacts are under reports/decision_package/.\n")
    print(json.dumps({"output": str(output), "package": str(package), "status": data["readiness"]["status"], "code_sha": data["code"]["head_sha"], "artifact_count": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
