#!/usr/bin/env python3
"""Run a guarded direct TemporalSigLIP Stage-A or Stage-B experiment.

This driver intentionally has no evidence/reranking path. It is prepared for
the next authorized GPU experiment but does not submit or authorize one.
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
from typing import Any, cast

import torch
from transformers import AutoProcessor

from qcpr_siglip2.data.loader import ExactBatch, make_exact_batches
from qcpr_siglip2.data.manifest import load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import RawFeatureBatch, build_relevance_masks, encode_real_features
from qcpr_siglip2.training.exposure import ExposureLedger
from qcpr_temporal_siglip.backbone import TemporalSigLIPBackbone
from qcpr_temporal_siglip.checkpointing import build_checkpoint
from qcpr_temporal_siglip.config import TemporalSigLIPConfig
from qcpr_temporal_siglip.model import TemporalSigLIP
from qcpr_temporal_siglip.objective import symmetric_mult_positive_clip_loss
from qcpr_temporal_siglip.trainer import build_optimizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sha256sums(run: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in run.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(run).as_posix()}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--physical-batch-size", type=int, default=32)
    parser.add_argument("--logical-physical-batch-size", type=int, default=128)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--authorize-long-run", action="store_true")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> int:
    expected = 512 if args.phase == "A" else 1536
    steps = expected if args.steps is None else int(args.steps)
    if steps != expected and os.environ.get("QCPR_ALLOW_NONSTANDARD_STEPS") != "1":
        raise ValueError(f"Phase {args.phase} requires exactly {expected} steps")
    if steps > 32 and not (
        args.authorize_long_run and os.environ.get("QCPR_ALLOW_LONG_TRAINING") == "1"
    ):
        raise RuntimeError("LONG_TRAINING_REQUIRES_EXPLICIT_AUTHORIZATION")
    return steps


def select_batch(
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


def encode_chunks(
    backbone: TemporalSigLIPBackbone,
    processor: Any,
    batch: ExactBatch,
    *,
    physical_batch_size: int,
    device: torch.device,
    no_grad: bool,
) -> RawFeatureBatch:
    if len(batch.pair_rows) % physical_batch_size:
        raise ValueError("logical pair batch must divide physical microbatch")
    chunks: list[RawFeatureBatch] = []
    captions_per_pair = len(batch.query_rows) // len(batch.pair_rows)
    for start in range(0, len(batch.pair_rows), physical_batch_size):
        end = start + physical_batch_size
        chunks.append(
            encode_real_features(
                backbone,
                processor,
                batch.pair_rows[start:end],
                batch.query_rows[start * captions_per_pair : end * captions_per_pair],
                device,
                dtype=torch.bfloat16,
                no_grad=no_grad,
            )
        )
    return RawFeatureBatch(
        frame_tokens=torch.cat([chunk.frame_tokens for chunk in chunks]),
        frame_embeddings=torch.cat([chunk.frame_embeddings for chunk in chunks]),
        text_tokens=torch.cat([chunk.text_tokens for chunk in chunks]),
        text_embeddings=torch.cat([chunk.text_embeddings for chunk in chunks]),
        text_mask=torch.cat([chunk.text_mask for chunk in chunks]),
    )


def logical_step(
    model: TemporalSigLIP,
    backbone: TemporalSigLIPBackbone,
    processor: Any,
    batch: ExactBatch,
    positive: torch.Tensor,
    ignored: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    physical_batch_size: int,
    device: torch.device,
    stage_b: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    frozen = encode_chunks(
        backbone,
        processor,
        batch,
        physical_batch_size=physical_batch_size,
        device=device,
        no_grad=True,
    )
    endpoints = RawFeatureBatch(
        frame_tokens=frozen.frame_tokens.detach().requires_grad_(stage_b),
        frame_embeddings=frozen.frame_embeddings.detach().requires_grad_(stage_b),
        text_tokens=frozen.text_tokens.detach().requires_grad_(stage_b),
        text_embeddings=frozen.text_embeddings.detach().requires_grad_(stage_b),
        text_mask=frozen.text_mask.detach(),
    )
    output = model.forward_from_features(
        endpoints.frame_tokens,
        endpoints.text_tokens,
        endpoints.text_embeddings,
        endpoints.text_mask,
    )
    loss = symmetric_mult_positive_clip_loss(output.score_matrix, positive, ignored)
    loss.backward()
    endpoint_grads = (
        endpoints.frame_tokens.grad,
        endpoints.frame_embeddings.grad,
        endpoints.text_tokens.grad,
        endpoints.text_embeddings.grad,
    )
    if stage_b:
        if any(value is None for value in endpoint_grads):
            raise RuntimeError("FEATURE_ENDPOINT_NO_GRADIENT")
        frame_grad, frame_embedding_grad, text_grad, text_embedding_grad = (
            cast(torch.Tensor, value) for value in endpoint_grads
        )
        captions_per_pair = len(batch.query_rows) // len(batch.pair_rows)
        for start in range(0, len(batch.pair_rows), physical_batch_size):
            end = start + physical_batch_size
            micro = ExactBatch(
                pair_rows=batch.pair_rows[start:end],
                query_rows=batch.query_rows[start * captions_per_pair : end * captions_per_pair],
                pair_ids=batch.pair_ids[start:end],
                query_ids=batch.query_ids[start * captions_per_pair : end * captions_per_pair],
            )
            recomputed = encode_chunks(
                backbone,
                processor,
                micro,
                physical_batch_size=physical_batch_size,
                device=device,
                no_grad=False,
            )
            query_start = start * captions_per_pair
            query_end = end * captions_per_pair
            surrogate = (
                (recomputed.frame_tokens * frame_grad[start:end]).sum()
                + (recomputed.frame_embeddings * frame_embedding_grad[start:end]).sum()
                + (recomputed.text_tokens * text_grad[query_start:query_end]).sum()
                + (recomputed.text_embeddings * text_embedding_grad[query_start:query_end]).sum()
            )
            surrogate.backward()
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("TRAINABLE_MODULE_NO_GRADIENT")
    flat = torch.cat(gradients)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("NONFINITE_GRADIENT")
    gradient_norm = float(flat.norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "score_shape": list(output.score_matrix.shape),
        "gradient_norm_preclip": gradient_norm,
        "multi_positive_queries": int((positive.sum(dim=1) > 1).sum()),
        "pair_embedding_norm_mean": float(output.pair_embedding.detach().float().norm(dim=-1).mean()),
        "text_embedding_norm_mean": float(output.text_embedding.detach().float().norm(dim=-1).mean()),
        "stage_b_backbone_recomputed": stage_b,
    }


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    steps = resolve_steps(args)
    if args.phase == "B" and not args.initial_checkpoint:
        raise ValueError("Phase B requires the accepted Stage-A checkpoint")
    if args.logical_physical_batch_size % args.physical_batch_size:
        raise ValueError("logical batch must divide into physical microbatches")
    if args.captions_per_pair <= 0:
        raise ValueError("captions_per_pair must be positive")
    worktree = Path(__file__).resolve().parents[1]
    state = git_state(worktree)
    if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    for path in (args.data_release, args.train_manifest, args.development_manifest, args.siglip2_model):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.phase == "B":
        initial_checkpoint = cast(str, args.initial_checkpoint)
        if not Path(initial_checkpoint).is_file():
            raise FileNotFoundError(initial_checkpoint)
    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("TEMPORAL_SIGLIP_TRAINING_REQUIRES_CUDA")
        torch.set_float32_matmul_precision("high")
        train_rows = load_exact_pair_rows(args.train_manifest, split="train")
        development_rows = load_exact_pair_rows(args.development_manifest, split="development")
        processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
        config = TemporalSigLIPConfig().validate()
        backbone = TemporalSigLIPBackbone(
            args.siglip2_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model = TemporalSigLIP(backbone, config).to(device)
        global_step_start = 0
        if args.phase == "B":
            initial_checkpoint = cast(str, args.initial_checkpoint)
            payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "model_state" not in payload:
                raise ValueError("invalid Stage-A checkpoint")
            model.set_stage_b()
            model.load_state_dict(payload["model_state"], strict=True)
            global_step_start = int(payload.get("global_step", 512))
            if global_step_start != 512:
                raise ValueError("Stage B must start at global step 512")
        else:
            model.set_stage_a()
        optimizer, scheduler, optimizer_report = build_optimizer(
            model,
            stage=args.phase,
            total_steps=steps,
        )
        write_json(run / "code_state.json", {"expected_code_sha": args.expected_code_sha, "runtime_git": state})
        write_json(run / "model_contract.json", {
            "model": "TemporalSigLIP",
            "primary_score": "exp(clamped_logit_scale) * cosine(text_embedding, pair_embedding)",
            "active_evidence_branch": False,
            "mandatory_reranking": False,
            "config": config.to_dict(),
        })
        write_json(run / "parameter_groups.json", optimizer_report)
        write_json(run / "config_resolved.json", {
            "phase": args.phase,
            "steps_this_phase": steps,
            "global_step_start": global_step_start,
            "global_step_end": global_step_start + steps,
            "physical_microbatch": args.physical_batch_size,
            "logical_physical_batch": args.logical_physical_batch_size,
            "logical_query_count": args.logical_physical_batch_size * args.captions_per_pair,
            "captions_per_pair": args.captions_per_pair,
            "seed": args.seed,
            "precision": "bf16",
            "early_stopping": False,
            "hard_negative_mining": False,
            "initial_checkpoint": args.initial_checkpoint,
            "optimizer_resumed": False,
        })
        write_json(run / "data_contract.json", {
            "data_release": args.data_release,
            "data_release_sha256": sha256_path(Path(args.data_release)),
            "train_manifest": args.train_manifest,
            "train_manifest_sha256": sha256(Path(args.train_manifest)),
            "development_manifest": args.development_manifest,
            "development_manifest_sha256": sha256(Path(args.development_manifest)),
            "train_pair_ids_sha256": ordered_id_sha256(train_rows, "canonical_pair_id"),
            "development_query_ids_sha256": ordered_id_sha256(development_rows, "caption_id"),
            "mask_access": False,
            "generated_unverified_text": False,
        })
        write_json(run / "batch_contract.json", {
            "physical_microbatch": args.physical_batch_size,
            "logical_physical_batch": args.logical_physical_batch_size,
            "captions_per_pair": args.captions_per_pair,
            "logical_score_matrix": [
                args.logical_physical_batch_size * args.captions_per_pair,
                args.logical_physical_batch_size,
            ],
            "feature_recompute_microbatches": args.logical_physical_batch_size // args.physical_batch_size,
            "one_symmetric_clip_loss": True,
        })
        write_json(run / "environment.json", {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "transformers": __import__("transformers").__version__,
            "cpu_peak_rss_gib_before": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        })
        ledger = ExposureLedger()
        metrics: list[dict[str, Any]] = []
        global_step = global_step_start
        total_start = time.perf_counter()
        milestone_steps = {0, 256, 512} if args.phase == "A" else {512, 1024, 2048}
        for local_step in range(steps):
            batch, epoch, batch_index = select_batch(
                train_rows,
                logical_size=args.logical_physical_batch_size,
                captions_per_pair=args.captions_per_pair,
                step=local_step + global_step_start,
                seed=args.seed,
            )
            positive, ignored, _ = build_relevance_masks(batch.query_rows, batch.pair_rows, device)
            step_result = logical_step(
                model,
                backbone,
                processor,
                batch,
                positive,
                ignored,
                optimizer,
                physical_batch_size=args.physical_batch_size,
                device=device,
                stage_b=args.phase == "B",
                scheduler=scheduler,
            )
            global_step += 1
            ledger.record_step(batch.pair_ids, batch.query_ids)
            metrics.append({"global_step": global_step, "epoch": epoch, "batch_index": batch_index, **step_result})
            if global_step in milestone_steps:
                torch.save(
                    build_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step=global_step,
                        metadata={"phase": args.phase, "code_sha": args.expected_code_sha},
                    ),
                    run / f"checkpoint_step_{global_step}.pt",
                )
        if global_step != global_step_start + steps:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")
        checkpoint = run / "checkpoint.pt"
        torch.save(
            build_checkpoint(
                model,
                optimizer,
                scheduler,
                global_step=global_step,
                metadata={"phase": args.phase, "code_sha": args.expected_code_sha},
            ),
            checkpoint,
        )
        write_json(run / "exposure_accounting.json", ledger.to_dict())
        write_json(run / "metrics.json", {"rows": metrics})
        write_json(run / "training_complete.json", {
            "status": "COMPLETED",
            "global_step": global_step,
            "requested_steps": steps,
            "global_step_start": global_step_start,
            "global_step_end": global_step_start + steps,
            "phase": args.phase,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "wall_seconds": time.perf_counter() - total_start,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        })
        write_sha256sums(run)
        return 0
    except Exception as exc:
        write_json(run / "failure.json", {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "python": sys.version,
        })
        write_sha256sums(run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
