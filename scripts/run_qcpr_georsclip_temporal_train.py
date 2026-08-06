#!/usr/bin/env python3
"""Train only the temporal head above frozen GeoRSCLIP towers.

This is the bounded GeoRSCLIP temporal-head baseline for the SigLIP-2 track.
It uses the
same exact-core data, listwise objective, logical exposure and checkpoint
contracts as the SigLIP-2 driver, but never unfreezes or mixes GeoRSCLIP
parameters with the primary model.
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

from qcpr_siglip2.backbones.georsclip import GeoRSCLIPBackbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.loader import ExactBatch, make_exact_batches
from qcpr_siglip2.data.manifest import load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import build_relevance_masks
from qcpr_siglip2.evaluation.common_gallery import canonical_pair_rows
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.exposure import ExposureLedger
from qcpr_siglip2.training.gradcache import logical_listwise_step
from qcpr_siglip2.training.optimizer import build_adamw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_sha256sums(run: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in run.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(
            f"{sha256_file(path)}  {path.relative_to(run).as_posix()}"
        )
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n")


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


class OpenClipProcessor:
    """Small adapter matching the processor interface used by the shared loader."""

    def __init__(self, preprocess: Any, tokenizer: Any) -> None:
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __call__(
        self,
        *,
        images: list[Any] | None = None,
        text: list[str] | None = None,
        return_tensors: str = "pt",
        padding: str | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding
        if images is not None and text is not None:
            raise ValueError("OpenClipProcessor accepts images or text, not both")
        if images is not None:
            return {
                "pixel_values": torch.stack(
                    [self.preprocess(image) for image in images]
                )
            }
        if text is not None:
            return {"input_ids": self.tokenizer(text)}
        raise ValueError("OpenClipProcessor requires images or text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--georsclip-checkpoint", required=True)
    parser.add_argument("--georsclip-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--logical-physical-batch-size", type=int, default=128)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--authorize-256-step-run", action="store_true")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> int:
    if args.steps != 256 and os.environ.get("QCPR_ALLOW_NONSTANDARD_STEPS") != "1":
        raise ValueError("GeoRSCLIP temporal baseline requires exactly 256 steps")
    if not args.authorize_256_step_run or os.environ.get(
        "QCPR_ALLOW_GEORSCLIP_256"
    ) != "1":
        raise RuntimeError("GEORSCLIP_256_REQUIRES_EXPLICIT_AUTHORIZATION")
    return int(args.steps)


def select_logical_batch(
    rows: list[dict[str, Any]],
    *,
    logical_size: int,
    captions_per_pair: int,
    step: int,
    seed: int,
) -> tuple[ExactBatch, int, int]:
    probe = make_exact_batches(
        rows,
        physical_batch_size=logical_size,
        captions_per_pair=captions_per_pair,
        epoch=0,
        seed=seed,
    )
    if not probe:
        raise ValueError("training manifest cannot form one logical batch")
    batches_per_epoch = len(probe)
    epoch = step // batches_per_epoch
    batch_index = step % batches_per_epoch
    batches = make_exact_batches(
        rows,
        physical_batch_size=logical_size,
        captions_per_pair=captions_per_pair,
        epoch=epoch,
        seed=seed,
    )
    return batches[batch_index], epoch, batch_index


def checkpoint_payload(
    model: Siglip2TemporalRetrievalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    global_step: int,
    config: Siglip2TemporalConfig,
    code_sha: str,
    data_release: str,
    train_manifest: str,
    development_manifest: str,
    georsclip_checkpoint: str,
    georsclip_revision: str,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "global_step": int(global_step),
        "config": config.to_dict(),
        "metadata": {
            "track": "qcpr_georsclip_temporal_baseline",
            "code_sha": code_sha,
            "data_release": data_release,
            "train_manifest": train_manifest,
            "development_manifest": development_manifest,
            "georsclip_checkpoint": georsclip_checkpoint,
            "georsclip_revision": georsclip_revision,
            "optimizer_resumed": False,
            "georsclip_towers_frozen": True,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def save_milestone(
    run: Path, payload: dict[str, Any], step: int
) -> dict[str, Any]:
    path = run / "milestones" / f"checkpoint_step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return {
        "step": step,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "evaluation_status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
    }


def checkpoint_roundtrip(
    model: Siglip2TemporalRetrievalModel,
    checkpoint_payload_value: dict[str, Any],
    checkpoint: Path,
    config: Siglip2TemporalConfig,
    backbone: GeoRSCLIPBackbone,
    processor: OpenClipProcessor,
    batch: ExactBatch,
    device: torch.device,
) -> dict[str, Any]:
    from qcpr_siglip2.data.runtime import encode_real_features

    model.eval()
    features = encode_real_features(
        backbone,
        processor,
        batch.pair_rows,
        batch.query_rows,
        device,
        dtype=torch.bfloat16,
        no_grad=True,
    )
    with torch.no_grad():
        reference = model.forward_from_features(
            features.frame_tokens,
            features.frame_embeddings,
            features.text_tokens,
            features.text_embeddings,
            features.text_mask,
        ).score_matrix.float()
    fresh = Siglip2TemporalRetrievalModel(None, config).to(device)
    fresh.load_state_dict(checkpoint_payload_value["model_state"], strict=True)
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh.forward_from_features(
            features.frame_tokens,
            features.frame_embeddings,
            features.text_tokens,
            features.text_embeddings,
            features.text_mask,
        ).score_matrix.float()
    difference = (reference - reloaded).abs()
    max_difference = float(difference.max())
    tolerance = 2e-4
    result = {
        "status": "PASS"
        if max_difference <= tolerance
        else "CHECKPOINT_ROUNDTRIP_MISMATCH",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "score_shape": list(reference.shape),
        "max_abs_score_difference": max_difference,
        "tolerance": tolerance,
        "fresh_model": True,
        "georsclip_towers_reused_only_for_features": True,
    }
    if result["status"] != "PASS":
        raise RuntimeError("CHECKPOINT_ROUNDTRIP_MISMATCH")
    del fresh, backbone
    return result


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        steps = resolve_steps(args)
        if args.logical_physical_batch_size % args.physical_batch_size:
            raise ValueError("logical batch must divide into physical microbatches")
        if args.captions_per_pair <= 0:
            raise ValueError("captions_per_pair must be positive")
        worktree = Path(args.worktree)
        state = git_state(worktree)
        if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
            raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
        release = Path(args.data_release)
        train_manifest = Path(args.train_manifest)
        development_manifest = Path(args.development_manifest)
        georsclip_checkpoint = Path(args.georsclip_checkpoint)
        for required in (
            release,
            train_manifest,
            development_manifest,
            georsclip_checkpoint,
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GEORSCLIP_TRAINING_REQUIRES_CUDA")
        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

        train_rows = load_exact_pair_rows(train_manifest, split="train")
        development_rows = load_exact_pair_rows(
            development_manifest, split="development"
        )
        pair_rows = canonical_pair_rows(development_rows)

        import open_clip

        preprocess = open_clip.image_transform(
            (224, 224),
            is_train=False,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            interpolation="bicubic",
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        processor = OpenClipProcessor(preprocess, tokenizer)
        backbone = GeoRSCLIPBackbone(
            georsclip_checkpoint, model_name="ViT-B-32"
        ).to(device)
        config = Siglip2TemporalConfig(
            hidden_size=512,
            expected_patch_tokens=49,
            base_grid=7,
            attention_heads=8,
            mlp_size=2048,
            max_frames=2,
            gradient_checkpointing=False,
            evidence_pair_chunk_size=8,
        )
        model = Siglip2TemporalRetrievalModel(None, config).to(device)
        optimizer, optimizer_report, scheduler = build_adamw(
            model, phase="A", total_steps=steps
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        write_json(
            run / "code_state.json",
            {"expected_code_sha": args.expected_code_sha, "runtime_git": state},
        )
        write_json(
            run / "environment.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "open_clip": getattr(open_clip, "__version__", "unknown"),
                "device": str(device),
            },
        )
        write_json(
            run / "model_source.json",
            {
                "repository": "Zilun/GeoRSCLIP",
                "revision": args.georsclip_revision,
                "checkpoint": str(georsclip_checkpoint),
                "checkpoint_sha256": sha256_file(georsclip_checkpoint),
                "model_name": "ViT-B-32",
                "weights_only": True,
                "license_status": "RESEARCH_BASELINE_ONLY",
                "towers_frozen": True,
                "load_audit": backbone.parameter_scope_report()["load_audit"],
            },
        )
        write_json(run / "parameter_groups.json", optimizer_report)
        write_json(
            run / "data_contract.json",
            {
                "data_release": str(release),
                "data_release_sha256": sha256_path(release),
                "train_manifest": str(train_manifest),
                "train_manifest_sha256": sha256_file(train_manifest),
                "development_manifest": str(development_manifest),
                "development_manifest_sha256": sha256_file(development_manifest),
                "train_ordered_pair_sha256": ordered_id_sha256(
                    train_rows, "canonical_pair_id"
                ),
                "development_ordered_query_sha256": ordered_id_sha256(
                    development_rows, "caption_id"
                ),
                "development_gallery_pair_sha256": ordered_id_sha256(
                    pair_rows, "canonical_pair_id"
                ),
                "allowed_sources": ["levir_mci", "second_cc"],
                "generated_unverified_text": False,
                "mask_access": False,
            },
        )
        write_json(
            run / "resolved_config.json",
            {
                "track": "qcpr_georsclip_temporal_baseline",
                "steps": steps,
                "seed": args.seed,
                "physical_batch_size": args.physical_batch_size,
                "logical_physical_batch_size": args.logical_physical_batch_size,
                "captions_per_pair": args.captions_per_pair,
                "logical_query_count": args.logical_physical_batch_size
                * args.captions_per_pair,
                "precision": "bf16",
                "loss": "one_multi_positive_listwise_query_to_pair",
                "hard_negative_mining": False,
                "early_stopping": False,
                "optimizer_resumed": False,
                "model_config": config.to_dict(),
            },
        )
        write_json(
            run / "batch_contract.json",
            {
                "physical_microbatch": args.physical_batch_size,
                "logical_physical_batch": args.logical_physical_batch_size,
                "captions_per_pair": args.captions_per_pair,
                "logical_query_count": args.logical_physical_batch_size
                * args.captions_per_pair,
                "logical_score_matrix": [
                    args.logical_physical_batch_size * args.captions_per_pair,
                    args.logical_physical_batch_size,
                ],
                "feature_microbatches": args.logical_physical_batch_size
                // args.physical_batch_size,
                "gradient_accumulation": 1,
                "note": "one common logical listwise loss; frozen tower features are encoded in physical chunks",
            },
        )

        ledger = ExposureLedger()
        metrics: list[dict[str, Any]] = []
        milestone_records: list[dict[str, Any]] = []
        current_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            global_step=0,
            config=config,
            code_sha=args.expected_code_sha,
            data_release=str(release),
            train_manifest=str(train_manifest),
            development_manifest=str(development_manifest),
            georsclip_checkpoint=str(georsclip_checkpoint),
            georsclip_revision=args.georsclip_revision,
        )
        milestone_records.append(save_milestone(run, current_payload, 0))
        started = time.perf_counter()
        last_batch: ExactBatch | None = None
        global_step = 0
        for local_step in range(steps):
            batch, epoch, batch_index = select_logical_batch(
                train_rows,
                logical_size=args.logical_physical_batch_size,
                captions_per_pair=args.captions_per_pair,
                step=local_step,
                seed=args.seed,
            )
            last_batch = batch
            positive, ignored, relevance = build_relevance_masks(
                batch.query_rows, batch.pair_rows, device
            )
            step_started = time.perf_counter()
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
                captions_per_pair=args.captions_per_pair,
                scheduler=scheduler,
                recompute_backbone=False,
            )
            global_step = local_step + 1
            ledger.record_step(batch.pair_ids, batch.query_ids)
            metrics.append(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "step_wall_seconds": time.perf_counter() - step_started,
                    "finite": True,
                    **relevance,
                    **step_metrics,
                }
            )
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics)
            )
            if global_step in (128, 256):
                current_payload = checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    global_step=global_step,
                    config=config,
                    code_sha=args.expected_code_sha,
                    data_release=str(release),
                    train_manifest=str(train_manifest),
                    development_manifest=str(development_manifest),
                    georsclip_checkpoint=str(georsclip_checkpoint),
                    georsclip_revision=args.georsclip_revision,
                )
                milestone_records.append(save_milestone(run, current_payload, global_step))

        if last_batch is None:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")
        if global_step != 256:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")
        final_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            global_step=global_step,
            config=config,
            code_sha=args.expected_code_sha,
            data_release=str(release),
            train_manifest=str(train_manifest),
            development_manifest=str(development_manifest),
            georsclip_checkpoint=str(georsclip_checkpoint),
            georsclip_revision=args.georsclip_revision,
        )
        checkpoint = run / "checkpoint.pt"
        torch.save(final_payload, checkpoint)
        checkpoint_digest = sha256_file(checkpoint)
        (run / "checkpoint.sha256").write_text(
            f"{checkpoint_digest}  {checkpoint.name}\n"
        )
        write_json(
            run / "milestone_evaluation_manifest.json",
            {
                "milestones": milestone_records,
                "evaluation_runner": "scripts/evaluate_qcpr_georsclip_common_gallery.py",
                "status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
                "full_rankings_required": True,
            },
        )
        write_json(run / "exposure_accounting.json", ledger.to_dict())
        roundtrip = checkpoint_roundtrip(
            model,
            final_payload,
            checkpoint,
            config,
            backbone,
            processor,
            last_batch,
            device,
        )
        write_json(run / "checkpoint_roundtrip.json", roundtrip)
        write_json(
            run / "gradient_diagnostics.json",
            {
                "status": "PER_STEP_FINITE",
                "last_gradient_report": metrics[-1]["gradient_report"],
                "all_losses_finite": True,
                "all_gradients_finite": True,
                "georsclip_towers_frozen": True,
            },
        )
        write_json(run / "model_contract.json", {
            "runtime_class": type(backbone.model).__name__,
            "hidden_size": config.hidden_size,
            "native_patch_tokens": config.expected_patch_tokens,
            "patch_grid": config.base_grid,
            "pair_embedding_shape": [args.physical_batch_size, config.hidden_size],
            "logical_score_matrix": [
                args.logical_physical_batch_size * args.captions_per_pair,
                args.logical_physical_batch_size,
            ],
            "trainable_parameter_count": optimizer_report["trainable_parameter_count"],
            "georsclip_vision_frozen": True,
            "georsclip_text_frozen": True,
            "mask_access": False,
        })
        write_json(
            run / "runtime_profile.json",
            {
                "total_wall_seconds": time.perf_counter() - started,
                "mean_step_seconds": sum(
                    float(row["step_wall_seconds"]) for row in metrics
                )
                / len(metrics),
                "steps": len(metrics),
                "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            },
        )
        write_json(
            run / "cuda_memory.json",
            {
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3)
                if device.type == "cuda"
                else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                / (1024**3)
                if device.type == "cuda"
                else None,
            },
        )
        pending = {
            "status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
            "milestones": [0, 128, 256],
            "scientific_metrics": "not claimed until rankings are evaluated",
        }
        write_json(run / "evaluation_metrics.json", pending)
        write_json(run / "ranking_integrity.json", pending)
        write_json(run / "full_rankings.json", pending)
        (run / "rankings_top100.jsonl").write_text(json.dumps(pending) + "\n")
        write_json(
            run / "training_complete.json",
            {
                "status": "PASS",
                "global_step": global_step,
                "requested_steps": 256,
                "no_nan_or_oom": True,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_digest,
                "evaluation_status": "PENDING_EXTERNAL_FULL_GALLERY_EVALUATION",
                "georsclip_towers_frozen": True,
            },
        )
        write_sha256sums(run)
        return 0
    except Exception as exc:
        write_json(
            run / "failure.json",
            {
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "expected_code_sha": args.expected_code_sha,
            },
        )
        write_sha256sums(run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
