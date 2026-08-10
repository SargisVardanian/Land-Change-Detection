#!/usr/bin/env python3
"""Run an auditable SigLIP-2 Phase-A or Phase-B logical-batch step.

This driver is deliberately guarded.  A run longer than 32 steps requires
both ``--authorize-long-run`` and ``QCPR_ALLOW_LONG_TRAINING=1``.  The current
audit task never sets either value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.loader import ExactBatch, make_unique_exact_batches
from qcpr_siglip2.data.manifest import load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import (
    PINNED_SIGLIP2_MAX_TEXT_LENGTH,
    _device_autocast,
    build_relevance_masks,
    encode_real_features,
    text_preprocessing_audit,
)
from qcpr_siglip2.data.resolution import use_partition_mode
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.exposure import ExposureLedger, sequence_sha256
from qcpr_siglip2.training.gradcache import logical_listwise_step
from qcpr_siglip2.training.milestones import (
    milestone_checkpoint_path,
    milestone_evaluation_path,
    required_milestones,
    steps_for_exposure,
)
from qcpr_siglip2.training.optimizer import build_adamw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory deterministically."""

    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(child)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_state(worktree: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(worktree), "branch", "--show-current"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"head": head, "branch": branch, "worktree_clean": not dirty}


def write_sha256sums(run: Path) -> None:
    lines: list[str] = []
    for path in sorted(path for path in run.rglob("*") if path.is_file()):
        if path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(run).as_posix()}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def failure_artifact(run: Path, exc: Exception) -> None:
    write_json(
        run / "failure.json",
        {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "python": sys.version,
        },
    )
    write_sha256sums(run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--siglip2-repository", required=True)
    parser.add_argument("--siglip2-revision", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--logical-physical-batch-size", type=int, default=128)
    parser.add_argument(
        "--max-unique-captions-per-pair", type=int, default=2
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--max-num-patches", type=int, required=True)
    parser.add_argument("--final-handoff", required=True)
    parser.add_argument("--tokenizer-gate", required=True)
    parser.add_argument("--multipositive-contract", required=True)
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=PINNED_SIGLIP2_MAX_TEXT_LENGTH,
    )
    parser.add_argument("--hierarchical-fraction", type=float, default=0.25)
    parser.add_argument("--authorize-long-run", action="store_true")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace, *, unique_pairs: int) -> int:
    phase_a_end = steps_for_exposure(
        8,
        unique_pairs=unique_pairs,
        logical_batch_size=args.logical_physical_batch_size,
    )
    total_end = steps_for_exposure(
        24,
        unique_pairs=unique_pairs,
        logical_batch_size=args.logical_physical_batch_size,
    )
    expected = phase_a_end if args.phase == "A" else total_end - phase_a_end
    steps = expected if args.steps is None else int(args.steps)
    if steps != expected and os.environ.get("QCPR_ALLOW_NONSTANDARD_STEPS") != "1":
        raise ValueError(
            f"Phase {args.phase} requires exactly {expected} steps unless the "
            "nonstandard test override is explicit"
        )
    if steps > 32 and not (
        args.authorize_long_run and os.environ.get("QCPR_ALLOW_LONG_TRAINING") == "1"
    ):
        raise RuntimeError("LONG_TRAINING_REQUIRES_EXPLICIT_AUTHORIZATION")
    return steps


def select_logical_batch(
    rows: list[dict[str, Any]],
    *,
    logical_size: int,
    max_unique_captions_per_pair: int,
    step: int,
    seed: int,
) -> tuple[ExactBatch, int, int]:
    """Rotate captions by epoch while keeping pair exposure deterministic."""

    probe = make_unique_exact_batches(
        rows,
        physical_batch_size=logical_size,
        max_unique_captions_per_pair=max_unique_captions_per_pair,
        epoch=0,
        seed=seed,
    )
    if not probe:
        raise ValueError("training manifest cannot form one complete logical batch")
    batches_per_epoch = len(probe)
    epoch = step // batches_per_epoch
    batch_index = step % batches_per_epoch
    batches = make_unique_exact_batches(
        rows,
        physical_batch_size=logical_size,
        max_unique_captions_per_pair=max_unique_captions_per_pair,
        epoch=epoch,
        seed=seed,
    )
    return batches[batch_index], epoch, batch_index


def load_checkpoint_weights(
    model: Siglip2TemporalRetrievalModel, path: str | None
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("initial checkpoint lacks model_state")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def checkpoint_payload(
    model: Siglip2TemporalRetrievalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    global_step: int,
    phase: str,
    code_sha: str,
    data_release: str,
    train_manifest: str,
    development_manifest: str,
) -> dict[str, Any]:
    """Build a complete, fresh-resume-safe checkpoint payload."""

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "global_step": global_step,
        "metadata": {
            "phase": phase,
            "global_step": global_step,
            "code_sha": code_sha,
            "data_release": data_release,
            "train_manifest": train_manifest,
            "development_manifest": development_manifest,
            "optimizer_resumed": False,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def save_milestone_checkpoint(
    run: Path,
    payload: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Save one checkpoint consumed by the external full-gallery evaluator."""

    checkpoint = milestone_checkpoint_path(run, step)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    digest = sha256(checkpoint)
    (checkpoint.with_name(checkpoint.name + ".sha256")).write_text(
        f"{digest}  {checkpoint.name}\n"
    )
    return {
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "evaluation_dir": str(milestone_evaluation_path(run, step)),
        "evaluation_status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
    }


def checkpoint_roundtrip(
    model: Siglip2TemporalRetrievalModel,
    backbone: Siglip2Backbone,
    processor: Any,
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
    model_config: Siglip2TemporalConfig,
    siglip2_model: str,
    phase: str,
    batch: ExactBatch,
    device: torch.device,
    max_num_patches: int,
) -> dict[str, Any]:
    """Reload the saved model in a fresh model/backbone and compare scores."""

    model.eval()
    features = encode_real_features(
        backbone,
        processor,
        batch.pair_rows,
        batch.query_rows,
        device,
        dtype=torch.bfloat16,
        no_grad=True,
        max_num_patches=max_num_patches,
    )
    with torch.no_grad(), _device_autocast(device, torch.bfloat16):
        reference = model.forward_from_features(
            features.frame_tokens,
            features.frame_embeddings,
            features.text_tokens,
            features.text_embeddings,
            features.text_mask,
            patch_valid_mask=features.patch_valid_mask,
            spatial_shapes=features.spatial_shapes,
            native_image_size=features.native_image_size,
            processed_patch_grid=features.processed_patch_grid,
            transform_hash=features.transform_hash,
            token_coordinates=features.token_coordinates,
        ).score_matrix.float()

    fresh_backbone = Siglip2Backbone(
        siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
    )
    if phase == "B":
        fresh_backbone.enable_phase_b_top_blocks(
            2, gradient_checkpointing=model_config.gradient_checkpointing
        )
    fresh_model = Siglip2TemporalRetrievalModel(
        fresh_backbone, model_config
    ).to(device)
    fresh_model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    fresh_model.eval()
    fresh_features = encode_real_features(
        fresh_backbone,
        processor,
        batch.pair_rows,
        batch.query_rows,
        device,
        dtype=torch.bfloat16,
        no_grad=True,
        max_num_patches=max_num_patches,
    )
    with torch.no_grad(), _device_autocast(device, torch.bfloat16):
        reloaded = fresh_model.forward_from_features(
            fresh_features.frame_tokens,
            fresh_features.frame_embeddings,
            fresh_features.text_tokens,
            fresh_features.text_embeddings,
            fresh_features.text_mask,
            patch_valid_mask=fresh_features.patch_valid_mask,
            spatial_shapes=fresh_features.spatial_shapes,
            native_image_size=fresh_features.native_image_size,
            processed_patch_grid=fresh_features.processed_patch_grid,
            transform_hash=fresh_features.transform_hash,
            token_coordinates=fresh_features.token_coordinates,
        ).score_matrix.float()
    difference = (reference - reloaded).abs()
    max_difference = float(difference.max())
    tolerance = 2e-4
    result = {
        "status": "PASS" if max_difference <= tolerance else "CHECKPOINT_ROUNDTRIP_MISMATCH",
        "fresh_model": True,
        "checkpoint": str(checkpoint),
        "score_shape": list(reference.shape),
        "max_abs_score_difference": max_difference,
        "tolerance": tolerance,
    }
    if result["status"] != "PASS":
        raise RuntimeError("CHECKPOINT_ROUNDTRIP_MISMATCH")
    del fresh_model, fresh_backbone, fresh_features, features
    return result


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "B" and not args.initial_checkpoint:
        raise ValueError("Phase B requires --initial-checkpoint from valid Phase A")
    if args.logical_physical_batch_size % args.physical_batch_size:
        raise ValueError("logical batch must divide into physical microbatches")
    if args.max_unique_captions_per_pair <= 0:
        raise ValueError("max_unique_captions_per_pair must be positive")
    worktree = Path(__file__).resolve().parents[1]
    state = git_state(worktree)
    if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    for required in (args.data_release, args.train_manifest, args.development_manifest):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    handoff_path = Path(args.final_handoff)
    tokenizer_gate_path = Path(args.tokenizer_gate)
    multipositive_path = Path(args.multipositive_contract)
    for required in (handoff_path, tokenizer_gate_path, multipositive_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not (
        handoff.get("EXACT_TRAINING_READY") is True
        and handoff.get("AUTHORIZE_TEMPORALSIGLIP_CORE_TRAINING") is True
        and handoff.get("main_training_allowed") is False
    ):
        raise RuntimeError("R19G_EXACT_CORE_AUTHORIZATION_MISMATCH")
    if Path(str(handoff.get("final_exact_train_manifest"))) != Path(
        args.train_manifest
    ):
        raise RuntimeError("R19G_TRAIN_MANIFEST_MISMATCH")
    tokenizer_gate = json.loads(tokenizer_gate_path.read_text(encoding="utf-8"))
    naflex_gate = tokenizer_gate.get("models", {}).get("naflex_256", {})
    if not (
        tokenizer_gate.get("status") == "PASS"
        and naflex_gate.get("status") == "PASS"
        and naflex_gate.get("repository") == args.siglip2_repository
        and naflex_gate.get("revision") == args.siglip2_revision
        and naflex_gate.get("actual_batch", {}).get(
            "all_valid_ids_in_embedding_range"
        )
        is True
    ):
        raise RuntimeError("TOKENIZER_CONFIG_GATE_FAILED")
    multipositive = json.loads(multipositive_path.read_text(encoding="utf-8"))
    if multipositive.get("MULTIPOSITIVE_SMOKE_READY") is not True:
        raise RuntimeError("MULTIPOSITIVE_CONTRACT_NOT_READY")
    if args.max_num_patches not in (256, 576, 1024):
        raise ValueError("max_num_patches must be an audited NaFlex budget")
    if args.max_text_length != PINNED_SIGLIP2_MAX_TEXT_LENGTH:
        raise ValueError(
            "the pinned SigLIP2 calibration policy requires max_text_length=64"
        )
    if not 0.0 <= args.hierarchical_fraction <= 1.0:
        raise ValueError("hierarchical_fraction must be in [0, 1]")
    config_path = Path(args.config_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not Path(args.siglip2_model).is_dir():
        raise FileNotFoundError(args.siglip2_model)
    steps_start = 0
    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("PHASE_TRAINING_REQUIRES_CUDA")
        torch.set_float32_matmul_precision("high")
        train_rows = load_exact_pair_rows(args.train_manifest, split="train")
        unique_pair_count = len(
            {str(row["canonical_pair_id"]) for row in train_rows}
        )
        manifest_query_ids = [str(row["caption_id"]) for row in train_rows]
        manifest_query_counts: dict[str, int] = {}
        for query_id in manifest_query_ids:
            manifest_query_counts[query_id] = manifest_query_counts.get(query_id, 0) + 1
        duplicate_manifest_query_ids = sorted(
            query_id for query_id, count in manifest_query_counts.items() if count > 1
        )
        if duplicate_manifest_query_ids:
            raise RuntimeError("QUERY_ID_MANIFEST_DUPLICATE")
        steps = resolve_steps(args, unique_pairs=unique_pair_count)
        development_rows = load_exact_pair_rows(
            args.development_manifest, split="development"
        )
        processor = AutoProcessor.from_pretrained(
            args.siglip2_model, local_files_only=True
        )
        write_json(
            run / "text_preprocessing_audit.json",
            {
                "policy": "pinned_siglip2_processor_v1",
                "train": text_preprocessing_audit(
                    processor,
                    train_rows,
                    max_text_length=args.max_text_length,
                ),
                "development": text_preprocessing_audit(
                    processor,
                    development_rows,
                    max_text_length=args.max_text_length,
                ),
            },
        )
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("config must contain a JSON object")
        config_fields = {
            name: config_payload[name]
            for name in Siglip2TemporalConfig.__dataclass_fields__
            if name in config_payload
        }
        model_config = Siglip2TemporalConfig.from_dict(config_fields)
        backbone = Siglip2Backbone(
            args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
        )
        if not backbone.is_naflex:
            raise RuntimeError("LEGACY_FIXRES_256PX_IS_EVALUATION_ONLY")
        model = Siglip2TemporalRetrievalModel(
            backbone, model_config
        ).to(device)
        caption_counts: dict[str, int] = {}
        for row in train_rows:
            pair_id = str(row["canonical_pair_id"])
            caption_counts[pair_id] = caption_counts.get(pair_id, 0) + 1
        singleton_pairs = sum(count == 1 for count in caption_counts.values())
        initial_payload = None
        if args.phase == "B":
            assert model.backbone is not None
            model.backbone.enable_phase_b_top_blocks(
                2, gradient_checkpointing=model_config.gradient_checkpointing
            )
            initial_payload = load_checkpoint_weights(model, args.initial_checkpoint)
            phase_a_end = steps_for_exposure(
                8,
                unique_pairs=unique_pair_count,
                logical_batch_size=args.logical_physical_batch_size,
            )
            steps_start = (
                int(initial_payload.get("global_step", phase_a_end))
                if initial_payload
                else phase_a_end
            )
            if steps_start != phase_a_end:
                raise ValueError(
                    "Phase B must start from the dynamic 8-presentations-per-pair "
                    "Phase-A checkpoint"
                )
        optimizer, optimizer_report, scheduler = build_adamw(
            model, phase=args.phase, total_steps=steps
        )
        milestone_steps = required_milestones(
            args.phase,
            unique_pairs=unique_pair_count,
            logical_batch_size=args.logical_physical_batch_size,
        )
        phase_end = steps_start + steps
        if not all(steps_start <= milestone <= phase_end for milestone in milestone_steps):
            raise RuntimeError("MILESTONE_CONTRACT_NOT_COVERED")
        write_json(run / "parameter_groups.json", optimizer_report)
        write_json(
            run / "code_state.json",
            {"expected_code_sha": args.expected_code_sha, "runtime_git": state},
        )
        write_json(
            run / "model_source.json",
            {
                "repository": args.siglip2_repository,
                "revision": args.siglip2_revision,
                "local_path": args.siglip2_model,
                "weights_sha256": sha256(Path(args.siglip2_model) / "model.safetensors"),
                "runtime_class": backbone.runtime_class,
                "vision_class": type(backbone.vision_model).__name__,
                "vision_config_class": type(backbone.vision_model.config).__name__,
                "text_class": type(backbone.text_model).__name__,
                "text_config_class": type(backbone.text_model.config).__name__,
                "processor_class": type(processor).__name__,
                "image_processor_class": type(processor.image_processor).__name__,
                "tokenizer_class": type(processor.tokenizer).__name__,
                "parameter_name_prefixes": [
                    "backbone.vision_model.",
                    "backbone.text_model.",
                    "temporal_adapter.",
                    "log_temperature",
                ],
                "legacy_encoder_contribution": False,
                "local_files_only": True,
            },
        )
        write_json(
            run / "environment.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "transformers": __import__("transformers").__version__,
            },
        )
        write_json(
            run / "resolved_config.json",
            {
                "phase": args.phase,
                "steps": steps,
                "global_step_start": steps_start,
                "global_step_end": steps_start + steps,
                "physical_batch_size": args.physical_batch_size,
                "logical_physical_batch_size": args.logical_physical_batch_size,
                "logical_query_count": "variable",
                "max_unique_captions_per_pair": args.max_unique_captions_per_pair,
                "variable_query_sampling": True,
                "caption_sampling": {
                    "policy": "deterministic_unique_rotation_without_replacement",
                    "max_unique_captions_per_pair": args.max_unique_captions_per_pair,
                    "singleton_pairs": singleton_pairs,
                    "singleton_caption_reuse": False,
                },
                "seed": args.seed,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "model_config": model_config.to_dict(),
                "precision": "bf16",
                "hard_negative_mining": False,
                "early_stopping": False,
                "initial_checkpoint": args.initial_checkpoint,
                "optimizer_resumed": False,
                "required_evaluation_milestones": list(milestone_steps),
                "unique_training_physical_pairs": unique_pair_count,
                "training_depth_unit": "presentations_per_unique_physical_pair",
                "max_num_patches": args.max_num_patches,
                "patch_budget_name": f"NAFLEX_{args.max_num_patches}_PATCH_BUDGET",
                "max_text_length": args.max_text_length,
                "hierarchical_fraction": args.hierarchical_fraction,
                "representation_modes": [
                    "DIRECT_NAFLEX",
                    "HIERARCHICAL_NATIVE",
                ],
                "same_retrieval_objective_for_representation_modes": True,
                "tokenizer_gate": str(tokenizer_gate_path),
                "tokenizer_gate_sha256": sha256(tokenizer_gate_path),
                "multipositive_contract": str(multipositive_path),
                "multipositive_contract_sha256": sha256(multipositive_path),
                "final_handoff": str(handoff_path),
                "final_handoff_sha256": sha256(handoff_path),
            },
        )
        write_json(
            run / "data_contract.json",
            {
                "release": args.data_release,
                "release_sha256": sha256_path(Path(args.data_release)),
                "train_manifest": args.train_manifest,
                "train_manifest_sha256": sha256(Path(args.train_manifest)),
                "development_manifest": args.development_manifest,
                "development_manifest_sha256": sha256(Path(args.development_manifest)),
                "train_ordered_pair_sha256": ordered_id_sha256(
                    train_rows, "canonical_pair_id"
                ),
                "development_ordered_query_sha256": ordered_id_sha256(
                    development_rows, "caption_id"
                ),
                "allowed_sources": ["levir_mci", "second_cc"],
                "mask_access": False,
                "generated_unverified_text": False,
                "exact_training_ready": True,
                "expanded_training_allowed": False,
                "final_handoff_sha256": sha256(handoff_path),
            },
        )
        write_json(
            run / "batch_contract.json",
            {
                "physical_microbatch": args.physical_batch_size,
                "logical_physical_batch": args.logical_physical_batch_size,
                "logical_query_count": "variable",
                "max_unique_captions_per_pair": args.max_unique_captions_per_pair,
                "logical_score_matrix": ["Q", args.logical_physical_batch_size],
                "query_ids_unique_within_logical_batch": True,
                "text_to_pair_reduction": "mean_query_loss_within_pair_then_mean_pairs",
                "pair_to_text_reduction": "multi_positive_unique_queries_then_mean_pairs",
                "feature_recompute_microbatches": args.logical_physical_batch_size
                // args.physical_batch_size,
                "gradient_accumulation": 1,
                "note": "one common logical listwise loss; physical chunks are feature-gradient replay, not independent losses",
            },
        )
        ledger = ExposureLedger()
        metric_rows: list[dict[str, Any]] = []
        sampling_schedule_rows: list[dict[str, Any]] = []
        milestone_records: dict[str, dict[str, Any]] = {}

        def save_current_milestone(step: int) -> None:
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                global_step=step,
                phase=args.phase,
                code_sha=args.expected_code_sha,
                data_release=args.data_release,
                train_manifest=args.train_manifest,
                development_manifest=args.development_manifest,
            )
            milestone_records[str(step)] = save_milestone_checkpoint(
                run, payload, step
            )

        total_start = time.perf_counter()
        global_step = steps_start
        last_batch: ExactBatch | None = None
        if global_step in milestone_steps:
            save_current_milestone(global_step)
        for local_step in range(steps):
            global_schedule_step = steps_start + local_step
            batch, epoch, batch_index = select_logical_batch(
                train_rows,
                logical_size=args.logical_physical_batch_size,
                max_unique_captions_per_pair=args.max_unique_captions_per_pair,
                step=global_schedule_step,
                seed=args.seed,
            )
            last_batch = batch
            positive, ignored, relevance = build_relevance_masks(
                batch.query_rows, batch.pair_rows, device
            )
            representation_mode = (
                "HIERARCHICAL_NATIVE"
                if use_partition_mode(
                    "logical_step",
                    epoch=global_schedule_step,
                    fraction=args.hierarchical_fraction,
                    seed=args.seed,
                )
                else "DIRECT_NAFLEX"
            )
            step_metrics = logical_listwise_step(
                model,
                backbone,
                processor,
                batch.pair_rows,
                batch.query_rows,
                positive,
                ignored,
                optimizer,
                device=device,
                physical_batch_size=args.physical_batch_size,
                query_pair_indices=batch.query_pair_indices,
                scheduler=scheduler,
                recompute_backbone=args.phase == "B",
                max_num_patches=args.max_num_patches,
                is_naflex=backbone.is_naflex,
                representation_mode=representation_mode,
                max_text_length=args.max_text_length,
            )
            ledger.record_step(
                batch.pair_ids,
                batch.query_ids,
                representation_mode=representation_mode,
            )
            sampling_schedule_rows.append(
                {
                    "global_step": global_step + 1,
                    "pair_ids": list(batch.pair_ids),
                    "query_ids": list(batch.query_ids),
                    "query_pair_indices": list(batch.query_pair_indices),
                    "captions_sampled_per_pair": [
                        batch.query_pair_indices.count(pair_index)
                        for pair_index in range(len(batch.pair_ids))
                    ],
                    "pair_sequence_sha256": sequence_sha256(batch.pair_ids),
                    "query_sequence_sha256": sequence_sha256(batch.query_ids),
                }
            )
            global_step += 1
            metric_rows.append(
                {
                    "global_step": global_step,
                    "local_step": local_step + 1,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "representation_mode": representation_mode,
                    **step_metrics,
                    **relevance,
                    "finite": True,
                }
            )
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows)
            )
            if global_step in milestone_steps:
                save_current_milestone(global_step)
        elapsed = time.perf_counter() - total_start
        final_step = global_step
        if final_step != steps_start + steps:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")
        if last_batch is None or len(last_batch.query_ids) != len(
            set(last_batch.query_ids)
        ):
            raise RuntimeError("QUERY_ID_DUPLICATE_IN_LOGICAL_BATCH")
        batch_contract = json.loads((run / "batch_contract.json").read_text())
        batch_contract.update(
            {
                "last_batch_pair_count": len(last_batch.pair_rows),
                "last_batch_query_count": len(last_batch.query_rows),
                "last_batch_score_matrix": [
                    len(last_batch.query_rows),
                    len(last_batch.pair_rows),
                ],
                "last_batch_pair_sequence_sha256": sequence_sha256(
                    last_batch.pair_ids
                ),
                "last_batch_query_sequence_sha256": sequence_sha256(
                    last_batch.query_ids
                ),
            }
        )
        write_json(run / "batch_contract.json", batch_contract)
        checkpoint_state = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            global_step=final_step,
            phase=args.phase,
            code_sha=args.expected_code_sha,
            data_release=args.data_release,
            train_manifest=args.train_manifest,
            development_manifest=args.development_manifest,
        )
        checkpoint = run / f"checkpoint_step_{final_step}.pt"
        torch.save(checkpoint_state, checkpoint)
        (run / "checkpoint.pt").write_bytes(checkpoint.read_bytes())
        (run / "checkpoint.sha256").write_text(
            f"{sha256(checkpoint)}  {checkpoint.name}\n"
        )
        write_json(
            run / "milestone_evaluation_manifest.json",
            {
                "phase": args.phase,
                "required_milestones": list(milestone_steps),
                "milestones": [milestone_records[str(step)] for step in milestone_steps],
                "evaluation_runner": "scripts/evaluate_qcpr_siglip2_milestones.py",
                "status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
                "full_rankings_required": True,
            },
        )
        schedule_path = run / "sampling_schedule.jsonl"
        schedule_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in sampling_schedule_rows
            ),
            encoding="utf-8",
        )
        exposure = ledger.to_dict()
        exposure["sampling_schedule_sha256"] = sha256(schedule_path)
        exposure["variable_query_count"] = True
        write_json(run / "exposure_accounting.json", exposure)
        sampled_query_ids = set(str(value) for value in ledger.query_sequence)
        write_json(
            run / "query_coverage_audit.json",
            {
                "manifest_row_count": len(manifest_query_ids),
                "manifest_unique_query_id_count": len(manifest_query_counts),
                "manifest_duplicate_query_ids": duplicate_manifest_query_ids,
                "manifest_query_id_gate": "PASS"
                if not duplicate_manifest_query_ids
                else "DATASET_INTEGRITY_FAILURE",
                "sampled_unique_query_ids": sorted(sampled_query_ids),
                "sampled_unique_query_id_count": len(sampled_query_ids),
                "never_sampled_query_ids": sorted(
                    set(manifest_query_counts) - sampled_query_ids
                ),
                "interpretation": (
                    "sampler coverage; manifest query IDs are unique"
                    if not duplicate_manifest_query_ids
                    else "dataset integrity failure"
                ),
            },
        )
        if last_batch is None:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")
        roundtrip = checkpoint_roundtrip(
            model,
            backbone,
            processor,
            checkpoint,
            checkpoint_state,
            model_config,
            args.siglip2_model,
            args.phase,
            last_batch,
            device,
            args.max_num_patches,
        )
        write_json(run / "checkpoint_roundtrip.json", roundtrip)
        write_json(
            run / "gradient_diagnostics.json",
            {
                "status": "PER_STEP_FINITE",
                "last_gradient_norm_preclip": metric_rows[-1]["gradient_norm_preclip"],
                "last_module_gradient_report": metric_rows[-1]["gradient_report"],
                "all_losses_finite": True,
                "all_gradients_finite": True,
            },
        )
        write_json(
            run / "runtime_profile.json",
            {
                "total_wall_seconds": elapsed,
                "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "steps_per_second": steps / max(elapsed, 1e-9),
            },
        )
        write_json(
            run / "cuda_memory.json",
            {
                "gpu": torch.cuda.get_device_name(device),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                / (1024**3),
                "current_allocated_gib": torch.cuda.memory_allocated(device)
                / (1024**3),
                "current_reserved_gib": torch.cuda.memory_reserved(device)
                / (1024**3),
            },
        )
        write_json(
            run / "embedding_diagnostics.json",
            metric_rows[-1]["embedding_diagnostics"],
        )
        write_json(
            run / "evidence_diagnostics.json",
            metric_rows[-1]["evidence_diagnostics"],
        )
        write_json(
            run / "model_contract.json",
            {
                "model_config": model_config.to_dict(),
                "runtime_class": backbone.runtime_class,
                "hidden_size": model_config.hidden_size,
                "expected_patch_tokens": model_config.expected_patch_tokens,
                "phase_b_top_blocks": backbone.phase_b_top_blocks,
                "gradient_checkpointing": {
                    "temporal": model_config.gradient_checkpointing,
                    "pretrained_top_blocks": backbone.gradient_checkpointing,
                },
            },
        )
        not_run = {
            "status": "PENDING_EXTERNAL_MILESTONE_EVALUATION",
            "reason": "run the guarded milestone evaluator before scientific acceptance",
            "required_milestones": list(milestone_steps),
        }
        write_json(run / "evaluation_metrics.json", not_run)
        write_json(run / "ranking_integrity.json", not_run)
        torch.save(not_run, run / "full_rankings.pt")
        (run / "rankings_top100.jsonl").write_text(json.dumps(not_run) + "\n")
        write_json(
            run / "training_complete.json",
            {
                "status": "PASS",
                "phase": args.phase,
                "global_step": final_step,
                "requested_steps": steps,
                "no_nan_or_oom": True,
                "checkpoint": str(checkpoint),
                "evaluation_status": "PENDING_EXTERNAL_MILESTONE_EVALUATION",
                "milestone_evaluation_manifest": str(
                    run / "milestone_evaluation_manifest.json"
                ),
            },
        )
        write_sha256sums(run)
        return 0
    except Exception as exc:
        failure_artifact(run, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
