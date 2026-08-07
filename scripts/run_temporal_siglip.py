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
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoProcessor

from qcpr_siglip2.data.loader import ExactBatch, make_exact_batches
from qcpr_siglip2.data.manifest import group_rows_by_pair, load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import RawFeatureBatch, build_relevance_masks, encode_real_features
from qcpr_siglip2.training.exposure import ExposureLedger
from qcpr_temporal_siglip.backbone import TemporalSigLIPBackbone
from qcpr_temporal_siglip.checkpointing import build_checkpoint
from qcpr_temporal_siglip.config import TemporalSigLIPConfig
from qcpr_temporal_siglip.localization import TemporalSoftChangeMap
from qcpr_temporal_siglip.model import TemporalSigLIP
from qcpr_temporal_siglip.objective import symmetric_mult_positive_clip_loss
from qcpr_temporal_siglip.trainer import build_optimizer
from validate_temporal_siglip_final_handoff import HandoffError, validate_handoff


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


def build_exposure_reports(
    ledger: ExposureLedger,
    pair_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize caption and source exposure without retaining caption text."""

    pair_by_id = {
        str(row["canonical_pair_id"]): row
        for group in group_rows_by_pair(pair_rows).values()
        for row in group[:1]
    }
    query_by_id = {str(row["caption_id"]): row for row in query_rows}
    source_presentations: Counter[str] = Counter()
    change_presentations: Counter[str] = Counter()
    for pair_id in ledger.pair_sequence:
        row = pair_by_id.get(str(pair_id), {})
        source = str(row.get("dataset_name") or row.get("source_dataset") or "unknown")
        source_presentations[source] += 1
        change_type = row.get("change_type") or row.get("change_category")
        change_presentations[str(change_type or "unknown")] += 1

    caption_presentations: Counter[str] = Counter(str(value) for value in ledger.query_sequence)
    verification_presentations: Counter[str] = Counter()
    scope_presentations: Counter[str] = Counter()
    for query_id in ledger.query_sequence:
        row = query_by_id.get(str(query_id), {})
        verification_presentations[str(row.get("verification") or "unknown")] += 1
        scope_presentations[str(row.get("query_scope") or "exact")] += 1

    caption_report = {
        "query_presentations": len(ledger.query_sequence),
        "unique_captions": len(caption_presentations),
        "presentations_by_caption_id": dict(sorted(caption_presentations.items())),
        "presentations_by_verification": dict(sorted(verification_presentations.items())),
        "presentations_by_query_scope": dict(sorted(scope_presentations.items())),
    }
    source_report = {
        "physical_pair_presentations": len(ledger.pair_sequence),
        "unique_physical_pairs": len(set(ledger.pair_sequence)),
        "presentations_by_source": dict(sorted(source_presentations.items())),
        "presentations_by_change_type": dict(sorted(change_presentations.items())),
    }
    return caption_report, source_report


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gradient_diagnostics(model: TemporalSigLIP) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        group_name = "backbone" if name.startswith("backbone.") else name.split(".", 1)[0]
        group = grouped.setdefault(
            group_name,
            {
                "parameter_count": 0,
                "trainable_count": 0,
                "parameters_with_grad": 0,
                "gradient_norm_squared": 0.0,
                "gradient_min": None,
                "gradient_max": None,
                "gradient_finite_elements": 0,
                "gradient_elements": 0,
            },
        )
        group["parameter_count"] += parameter.numel()
        if parameter.requires_grad:
            group["trainable_count"] += parameter.numel()
        if parameter.grad is None:
            continue
        group["parameters_with_grad"] += 1
        values = parameter.grad.detach().float()
        group["gradient_norm_squared"] += float(values.square().sum().cpu())
        group["gradient_finite_elements"] += int(torch.isfinite(values).sum().cpu())
        group["gradient_elements"] += values.numel()
        minimum = float(values.min().cpu())
        maximum = float(values.max().cpu())
        group["gradient_min"] = (
            minimum
            if group["gradient_min"] is None
            else min(float(group["gradient_min"]), minimum)
        )
        group["gradient_max"] = (
            maximum
            if group["gradient_max"] is None
            else max(float(group["gradient_max"]), maximum)
        )
    for group in grouped.values():
        group["gradient_norm"] = float(group.pop("gradient_norm_squared") ** 0.5)
        elements = int(group["gradient_elements"])
        group["finite_gradient_fraction"] = (
            float(group["gradient_finite_elements"] / elements) if elements else 1.0
        )
    return {"modules": grouped}


def checkpoint_roundtrip(
    checkpoint: Path,
    model: TemporalSigLIP,
    config: TemporalSigLIPConfig,
    frame_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        before = model.forward_from_features(
            frame_tokens,
            text_tokens,
            text_embeddings,
            text_mask,
        ).score_matrix.detach().float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("checkpoint roundtrip payload is invalid")
    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model_state is invalid")
    temporal_state = {
        key: value for key, value in model_state.items() if not key.startswith("backbone.")
    }
    reloaded = TemporalSigLIP(config=config).to(device)
    reloaded.load_state_dict(temporal_state, strict=True)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.forward_from_features(
            frame_tokens,
            text_tokens,
            text_embeddings,
            text_mask,
        ).score_matrix.detach().float()
    maximum_absolute_difference = float((before - after).abs().max().cpu())
    tolerance = 1e-5
    return {
        "status": "PASS" if maximum_absolute_difference <= tolerance else "FAIL",
        "tolerance": tolerance,
        "max_absolute_score_difference": maximum_absolute_difference,
        "backbone_state_keys_in_checkpoint": sum(
            1 for key in model_state if str(key).startswith("backbone.")
        ),
    }


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
    parser.add_argument(
        "--final-handoff",
        help="Require and validate the authoritative Dataset-Agent final handoff.",
    )
    parser.add_argument(
        "--target-pair-presentations",
        type=int,
        help="Record the declared unique-pair exposure target for a final run.",
    )
    parser.add_argument(
        "--max-pair-presentations",
        type=int,
        help="Fail if any physical pair exceeds this exposure ceiling.",
    )
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
    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    chunks: list[RawFeatureBatch] = []
    captions_per_pair = len(batch.query_rows) // len(batch.pair_rows)
    for start in range(0, len(batch.pair_rows), physical_batch_size):
        end = min(start + physical_batch_size, len(batch.pair_rows))
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
    step_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    synchronize_cuda(device)
    feature_start = time.perf_counter()
    frozen = encode_chunks(
        backbone,
        processor,
        batch,
        physical_batch_size=physical_batch_size,
        device=device,
        no_grad=True,
    )
    synchronize_cuda(device)
    feature_seconds = time.perf_counter() - feature_start
    endpoints = RawFeatureBatch(
        frame_tokens=frozen.frame_tokens.detach().requires_grad_(stage_b),
        frame_embeddings=frozen.frame_embeddings.detach().requires_grad_(stage_b),
        text_tokens=frozen.text_tokens.detach().requires_grad_(stage_b),
        text_embeddings=frozen.text_embeddings.detach().requires_grad_(stage_b),
        text_mask=frozen.text_mask.detach(),
    )
    forward_start = time.perf_counter()
    output = model.forward_from_features(
        endpoints.frame_tokens,
        endpoints.text_tokens,
        endpoints.text_embeddings,
        endpoints.text_mask,
    )
    loss = symmetric_mult_positive_clip_loss(output.score_matrix, positive, ignored)
    synchronize_cuda(device)
    forward_seconds = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    loss.backward()
    # The active direct model consumes native visual patch tokens and the
    # pooled text embedding.  Pooled visual features and text token features
    # are retained in RawFeatureBatch for the shared loader contract, but are
    # not inputs to TemporalSigLIP.forward_from_features and therefore must
    # not be treated as required Stage-B gradient endpoints.
    endpoint_grads = (endpoints.frame_tokens.grad, endpoints.text_embeddings.grad)
    if stage_b:
        if any(value is None for value in endpoint_grads):
            raise RuntimeError("FEATURE_ENDPOINT_NO_GRADIENT")
        frame_grad, text_embedding_grad = (
            cast(torch.Tensor, value) for value in endpoint_grads
        )
        captions_per_pair = len(batch.query_rows) // len(batch.pair_rows)
        for start in range(0, len(batch.pair_rows), physical_batch_size):
            end = min(start + physical_batch_size, len(batch.pair_rows))
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
                + (recomputed.text_embeddings * text_embedding_grad[query_start:query_end]).sum()
            )
            surrogate.backward()
    synchronize_cuda(device)
    backward_seconds = time.perf_counter() - backward_start
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
    optimizer_start = time.perf_counter()
    optimizer.step()
    scheduler.step()
    synchronize_cuda(device)
    optimizer_seconds = time.perf_counter() - optimizer_start
    return {
        "loss": float(loss.detach().cpu()),
        "score_shape": list(output.score_matrix.shape),
        "gradient_norm_preclip": gradient_norm,
        "multi_positive_queries": int((positive.sum(dim=1) > 1).sum()),
        "pair_embedding_norm_mean": float(output.pair_embedding.detach().float().norm(dim=-1).mean()),
        "text_embedding_norm_mean": float(output.text_embedding.detach().float().norm(dim=-1).mean()),
        "stage_b_backbone_recomputed": stage_b,
        "feature_seconds": feature_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_step_seconds": optimizer_seconds,
        "total_step_seconds": time.perf_counter() - step_start,
    }


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    steps = resolve_steps(args)
    if args.phase == "B" and not args.initial_checkpoint:
        raise ValueError("Phase B requires the accepted Stage-A checkpoint")
    if args.physical_batch_size > args.logical_physical_batch_size:
        raise ValueError("physical microbatch cannot exceed logical physical batch")
    if args.captions_per_pair <= 0:
        raise ValueError("captions_per_pair must be positive")
    if args.target_pair_presentations is not None and args.target_pair_presentations <= 0:
        raise ValueError("target_pair_presentations must be positive")
    if args.max_pair_presentations is not None and args.max_pair_presentations <= 0:
        raise ValueError("max_pair_presentations must be positive")
    if (
        args.target_pair_presentations is not None
        and args.max_pair_presentations is not None
        and args.target_pair_presentations > args.max_pair_presentations
    ):
        raise ValueError("target_pair_presentations cannot exceed max_pair_presentations")
    if steps <= 0:
        raise ValueError("steps must be positive")
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
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")
        train_rows = load_exact_pair_rows(args.train_manifest, split="train")
        development_rows = load_exact_pair_rows(args.development_manifest, split="development")
        final_handoff_result: dict[str, Any] | None = None
        if args.final_handoff:
            try:
                final_handoff_result = validate_handoff(
                    Path(args.final_handoff),
                    project_root=worktree,
                )
            except HandoffError:
                raise
            expected_train = Path(
                str(final_handoff_result["manifests"]["final_exact_train_manifest"]["path"])
            ).resolve()
            expected_development = Path(
                str(final_handoff_result["manifests"]["final_exact_development_manifest"]["path"])
            ).resolve()
            if Path(args.train_manifest).resolve() != expected_train:
                raise RuntimeError("FINAL_HANDOFF_TRAIN_MANIFEST_MISMATCH")
            if Path(args.development_manifest).resolve() != expected_development:
                raise RuntimeError("FINAL_HANDOFF_DEVELOPMENT_MANIFEST_MISMATCH")
            release_sha = str(final_handoff_result["authoritative_release_sha"])
            actual_release_sha = sha256_path(Path(args.data_release))
            if actual_release_sha != release_sha:
                raise RuntimeError("FINAL_HANDOFF_RELEASE_SHA_MISMATCH")
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
            "final_handoff": args.final_handoff,
            "target_pair_presentations": args.target_pair_presentations,
            "max_pair_presentations": args.max_pair_presentations,
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
            "final_handoff": args.final_handoff,
            "final_handoff_validation": final_handoff_result,
        })
        write_json(run / "batch_contract.json", {
            "physical_microbatch": args.physical_batch_size,
            "logical_physical_batch": args.logical_physical_batch_size,
            "captions_per_pair": args.captions_per_pair,
            "logical_score_matrix": [
                args.logical_physical_batch_size * args.captions_per_pair,
                args.logical_physical_batch_size,
            ],
            "feature_recompute_microbatches": (
                args.logical_physical_batch_size + args.physical_batch_size - 1
            ) // args.physical_batch_size,
            "feature_recompute_microbatch_sizes": [
                min(args.physical_batch_size, args.logical_physical_batch_size - start)
                for start in range(0, args.logical_physical_batch_size, args.physical_batch_size)
            ],
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
        last_batch: ExactBatch | None = None
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
            last_batch = batch
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
        pair_counts = Counter(str(value) for value in ledger.pair_sequence)
        if args.max_pair_presentations is not None:
            over_ceiling = {
                pair_id: count
                for pair_id, count in pair_counts.items()
                if count > args.max_pair_presentations
            }
            if over_ceiling:
                raise RuntimeError(
                    "PAIR_PRESENTATION_CEILING_EXCEEDED: "
                    + json.dumps(dict(sorted(over_ceiling.items())[:10]), sort_keys=True)
                )
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
        caption_exposure, source_exposure = build_exposure_reports(
            ledger,
            train_rows,
            train_rows,
        )
        write_json(run / "caption_exposure.json", caption_exposure)
        write_json(run / "source_exposure.json", source_exposure)
        write_json(run / "metrics.json", {"rows": metrics})
        with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if last_batch is None:
            raise RuntimeError("NO_TRAINING_BATCH_RECORDED")
        final_features = encode_chunks(
            backbone,
            processor,
            last_batch,
            physical_batch_size=args.physical_batch_size,
            device=device,
            no_grad=True,
        )
        roundtrip = checkpoint_roundtrip(
            checkpoint,
            model,
            config,
            final_features.frame_tokens,
            final_features.text_tokens,
            final_features.text_embeddings,
            final_features.text_mask,
            device,
        )
        write_json(run / "checkpoint_roundtrip.json", roundtrip)
        if roundtrip["status"] != "PASS":
            raise RuntimeError("CHECKPOINT_ROUNDTRIP_MISMATCH")
        gradient_report = gradient_diagnostics(model)
        write_json(run / "gradient_diagnostics.json", gradient_report)
        localizer = TemporalSoftChangeMap(config).to(device)
        model.eval()
        with torch.no_grad():
            final_output = model.forward_from_features(
                final_features.frame_tokens,
                final_features.text_tokens,
                final_features.text_embeddings,
                final_features.text_mask,
            )
            local_output = localizer(
                final_output.temporal_tokens[:2],
                final_output.text_embedding[:2],
            )
            swapped_output = localizer(
                final_output.temporal_tokens[:2],
                final_output.text_embedding[:2].flip(0),
            )
        map_l1 = float(
            (local_output.map_probabilities - swapped_output.map_probabilities)
            .abs()
            .mean()
            .cpu()
        )
        write_json(run / "localization_metrics.json", {
            "status": "DIAGNOSTIC_ONLY_NO_LOCALIZED_TRAINING_QUERIES",
            "map_shape": list(local_output.map_logits.shape),
            "query_swap_map_l1": map_l1,
            "trainable_localization_parameters": sum(
                parameter.numel() for parameter in localizer.parameters()
            ),
        })
        write_json(run / "evidence_diagnostics.json", {
            "primary_score_evidence_path": False,
            "status": "NOT_APPLICABLE_DIRECT_GLOBAL_MODEL",
            "post_retrieval_soft_map": True,
            "map_is_training_objective": False,
        })
        timing_rows = metrics
        average = lambda key: float(
            sum(float(row[key]) for row in timing_rows) / len(timing_rows)
        )
        write_json(run / "runtime_profile.json", {
            "steps": len(metrics),
            "seconds_per_forward": average("forward_seconds"),
            "seconds_per_backward": average("backward_seconds"),
            "seconds_per_optimizer_step": average("optimizer_step_seconds"),
            "seconds_per_feature_extraction": average("feature_seconds"),
            "seconds_per_total_step": average("total_step_seconds"),
            "number_of_visual_tokens": int(final_features.frame_tokens.shape[2]),
            "number_of_text_tokens": int(final_features.text_tokens.shape[1]),
            "score_matrix_shape": list(final_output.score_matrix.shape),
            "image_and_text_decode_included_in_feature_seconds": True,
        })
        write_json(run / "cuda_memory.json", {
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "current_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "current_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
        })
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
