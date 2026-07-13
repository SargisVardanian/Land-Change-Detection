from __future__ import annotations

import json
import math
import sys
import argparse
import hashlib
from pathlib import Path

from land_change_detection.training.temporal_caption_dataset import (
    TemporalCaptionManifestDataset,
    load_dataset_config,
)


def _file_fingerprint(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_weight(value: str) -> tuple[str, float]:
    name, raw = value.split("=", 1)
    return name, float(raw)


def _dataset_counts(dataset: TemporalCaptionManifestDataset) -> dict[str, int]:
    return {str(name): len(indices) for name, indices in sorted(dataset.indices_by_dataset.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_report", type=Path)
    parser.add_argument("memory_report", type=Path)
    parser.add_argument("current_commit")
    parser.add_argument("world_size", type=int)
    parser.add_argument("minimum_global_batch", type=int)
    parser.add_argument("expected_temporal_depth", type=int, nargs="?", default=6)
    parser.add_argument("--expected-data-mode", choices=("levir_only", "mixed"), default="levir_only")
    parser.add_argument("--expected-train-manifest", action="append", default=[])
    parser.add_argument("--expected-val-manifest", action="append", default=[])
    parser.add_argument("--expected-dataset-weight", action="append", default=[])
    parser.add_argument("--expected-dataset-config", type=Path, default=None)
    parser.add_argument("--expected-text-max-length", type=int, default=256)
    parser.add_argument("--expected-patch-reranker", action="store_true")
    return parser.parse_args()


def _expected_from_args(args: argparse.Namespace) -> dict[str, object]:
    train_manifests = [str(path) for path in args.expected_train_manifest]
    val_manifests = [str(path) for path in args.expected_val_manifest]
    weights = dict(_parse_weight(value) for value in args.expected_dataset_weight)
    allowed_caption_sources: set[str] | None = None
    if args.expected_dataset_config:
        config_train, config_val, config_weights, config_options = load_dataset_config(args.expected_dataset_config)
        train_manifests = config_train or train_manifests
        val_manifests = config_val or val_manifests
        weights = config_weights or weights
        if config_options.get("allowed_caption_sources"):
            allowed_caption_sources = {str(value) for value in config_options["allowed_caption_sources"]}
    full_train = TemporalCaptionManifestDataset(
        train_manifests,
        split="train",
        image_size=256,
        output_grid=32,
        max_pairs=None,
        allowed_caption_sources=allowed_caption_sources,
    )
    full_val = TemporalCaptionManifestDataset(
        val_manifests,
        split="val",
        image_size=256,
        output_grid=32,
        max_pairs=None,
        allowed_caption_sources=allowed_caption_sources,
    )
    selected_train = TemporalCaptionManifestDataset(
        train_manifests,
        split="train",
        image_size=256,
        output_grid=32,
        max_pairs=20,
        allowed_caption_sources=allowed_caption_sources,
    )
    selected_val = TemporalCaptionManifestDataset(
        val_manifests,
        split="val",
        image_size=256,
        output_grid=32,
        max_pairs=16,
        allowed_caption_sources=allowed_caption_sources,
    )
    dataset_names: set[str] = set()
    for dataset in (full_train, full_val):
        dataset_names.update(str(name) for name in dataset.indices_by_dataset)
    return {
        "train_manifests": train_manifests,
        "val_manifests": val_manifests,
        "manifest_fingerprints": {
            "train": {path: _file_fingerprint(path) for path in train_manifests},
            "validation": {path: _file_fingerprint(path) for path in val_manifests},
        },
        "dataset_names": sorted(dataset_names),
        "dataset_weights": weights if args.expected_data_mode == "mixed" else {},
        "train_row_count": len(full_train),
        "validation_row_count": len(full_val),
        "full_train_row_count": len(full_train),
        "full_validation_row_count": len(full_val),
        "selected_train_row_count": len(selected_train),
        "selected_validation_row_count": len(selected_val),
        "sample_counts_by_dataset": _dataset_counts(selected_train),
        "validation_sample_counts_by_dataset": _dataset_counts(selected_val),
    }


def main() -> int:
    args = parse_args()
    smoke_path = args.smoke_report
    memory_path = args.memory_report
    current_commit = args.current_commit
    world_size = args.world_size
    minimum_global_batch = args.minimum_global_batch
    expected_temporal_depth = args.expected_temporal_depth
    if not smoke_path.exists() or not memory_path.exists():
        raise SystemExit("Missing Stage-1-next smoke or memory report")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    smoke_checks = {
        "real_cluster_smoke_passed": True,
        "git_commit": current_commit,
        "steps_completed": 10,
        "finite_loss": True,
        "device_type": "cuda",
        "bf16_active": True,
        "fake_backbones": False,
        "train_val_disjoint": True,
        "frozen_grad_violations": [],
        "missing_gradients": [],
        "checkpoint_roundtrip_passed": True,
        "image_size": 256,
        "output_grid": 32,
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": expected_temporal_depth,
        "text_adapter_enabled": True,
        "text_max_length": args.expected_text_max_length,
    }
    if args.expected_patch_reranker:
        smoke_checks["patch_reranker_available"] = True
    errors = [
        f"smoke {key}={smoke.get(key)!r}"
        for key, expected in smoke_checks.items()
        if smoke.get(key) != expected
    ]
    if not smoke.get("slurm_job_id"):
        errors.append("smoke slurm_job_id is required for real cluster readiness")
    if not smoke.get("slurm_job_name"):
        errors.append("smoke slurm_job_name is required for real cluster readiness")
    for required_key in ("gpu_name", "device_type", "bf16_active", "image_size", "output_grid"):
        if required_key not in smoke:
            errors.append(f"smoke {required_key} is required")
    if "H100" not in smoke.get("gpu_name", ""):
        errors.append("Stage-1-next smoke did not run on H100")
    if smoke.get("data_mode", "levir_only") != args.expected_data_mode:
        errors.append(f"smoke data_mode={smoke.get('data_mode')!r}; expected {args.expected_data_mode!r}")
    if args.expected_data_mode == "mixed":
        expected = _expected_from_args(args)
        for key in (
            "manifest_fingerprints",
            "dataset_names",
            "dataset_weights",
            "train_row_count",
            "validation_row_count",
            "full_train_row_count",
            "full_validation_row_count",
            "selected_train_row_count",
            "selected_validation_row_count",
        ):
            if smoke.get(key) != expected[key]:
                errors.append(f"smoke {key}={smoke.get(key)!r}; expected {expected[key]!r}")
        if smoke.get("mixed_smoke") is not True:
            errors.append("mixed training requires a smoke report with mixed_smoke=true")
        if smoke.get("mixed_subset_coverage_passed") is not True:
            errors.append("mixed training requires mixed_subset_coverage_passed=true")
        if len(smoke.get("dataset_names", [])) < 2:
            errors.append("mixed smoke did not include at least two datasets")
        expected_count_keys = {
            "sample_counts_by_dataset": "selected_train_row_count",
            "validation_sample_counts_by_dataset": "selected_validation_row_count",
        }
        for key, total_key in expected_count_keys.items():
            counts = smoke.get(key, {})
            if not isinstance(counts, dict):
                errors.append(f"smoke {key}={counts!r}; expected dataset count mapping")
                continue
            if counts != expected[key]:
                errors.append(f"smoke {key}={counts!r}; expected {expected[key]!r}")
            if sum(int(value) for value in counts.values()) != expected[total_key]:
                errors.append(f"smoke {key} totals {counts!r}; expected selected row count {expected[total_key]!r}")
            for required_dataset in ("levir_mci", "second_cc"):
                if required_dataset in expected["dataset_names"] and int(counts.get(required_dataset, 0)) <= 0:
                    errors.append(f"mixed smoke selected {key} is missing {required_dataset}")
    memory_checks = {
        "status": "PASS",
        "git_commit": current_commit,
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": expected_temporal_depth,
        "text_adapter_enabled": True,
        "text_max_length": args.expected_text_max_length,
    }
    if args.expected_patch_reranker:
        memory_checks["patch_reranker_available"] = True
    errors.extend(
        f"memory {key}={memory.get(key)!r}"
        for key, expected in memory_checks.items()
        if memory.get(key) != expected
    )
    if memory.get("memory_data_mode") != "shape_probe":
        errors.append(f"memory memory_data_mode={memory.get('memory_data_mode')!r}; expected 'shape_probe'")
    if args.expected_patch_reranker:
        if smoke.get("qcpr_score_mode") != "fused":
            errors.append("patch-reranker smoke must report qcpr_score_mode='fused'")
        if smoke.get("qcpr_gradient_audit_passed") is not True:
            errors.append("patch-reranker smoke must pass QCPR gradient audit")
        if memory.get("qcpr_score_mode") != "fused":
            errors.append("patch-reranker memory probe must report qcpr_score_mode='fused'")
        if int(smoke.get("retrieval_supervised_queries", 0)) <= 0:
            errors.append("patch-reranker smoke must include retrieval-supervised queries")
        if int(smoke.get("segmentation_supervised_pairs", 0)) <= 0:
            errors.append("patch-reranker smoke must include segmentation-supervised pairs")
        if smoke.get("supervision_evidence_passed") is not True:
            errors.append("patch-reranker smoke must pass mixed-supervision evidence checks")
        if not isinstance(smoke.get("target_kind_counts"), dict):
            errors.append("patch-reranker smoke must report target source counts")
        if not isinstance(smoke.get("target_source_counts"), dict):
            errors.append("patch-reranker smoke must report segmentation target-source counts")
        for key in ("query_specific_segmentation_losses", "generic_change_segmentation_losses"):
            values = smoke.get(key)
            if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values):
                errors.append(f"patch-reranker smoke must report finite {key}")
        if args.expected_data_mode == "mixed" and "s2looking" in smoke.get("dataset_names", []):
            if int(smoke.get("sample_counts_by_dataset", {}).get("s2looking", 0)) <= 0:
                errors.append("mixed patch-reranker smoke must select S2Looking")
            if int(smoke.get("retrieval_supervised_pairs", 0)) >= int(smoke.get("total_pairs_seen", 0)):
                errors.append("mixed S2Looking smoke must exclude at least one pair from retrieval supervision")
            localization = smoke.get("localization_validation_smoke", {})
            if not isinstance(localization, dict) or localization.get("required") is not True:
                errors.append("mixed S2Looking smoke must require dedicated localization validation")
            elif localization.get("configured") is not True or localization.get("passed") is not True:
                errors.append("mixed S2Looking localization validation smoke must be configured and pass")
            localization_coverage = smoke.get("localization_validation_coverage", {})
            if not isinstance(localization_coverage, dict) or localization_coverage.get("expected_datasets") != ["s2looking"]:
                errors.append("localization validation must expect only S2Looking")
    recommended = int(memory.get("recommended_batch_size") or 0)
    required_local = math.ceil(minimum_global_batch / world_size)
    if recommended < required_local:
        errors.append(f"required local batch {required_local} did not fit; probe recommended {recommended}")
    if errors:
        raise SystemExit("UniChange v2 Stage-1-next readiness gate failed: " + "; ".join(errors))
    print(required_local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
