#!/usr/bin/env python3
"""Measure real SigLIP-2 partial-fine-tuning batch sizes on one H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.loader import ExactBatch, make_exact_batches
from qcpr_siglip2.data.manifest import load_exact_pair_rows
from qcpr_siglip2.data.runtime import build_relevance_masks, processor_inputs
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.gradcache import module_gradient_report
from qcpr_siglip2.training.objective import multi_positive_listwise_loss
from qcpr_siglip2.training.optimizer import build_adamw


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
    lines = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{sha256(path)}  {path.name}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_state(worktree: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    return {
        "head": head,
        "branch": subprocess.check_output(
            ["git", "-C", str(worktree), "branch", "--show-current"], text=True
        ).strip(),
        "worktree_clean": not dirty,
    }


def peak_cpu_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def candidate_batch(
    rows: list[dict[str, Any]], physical_batch_size: int, captions_per_pair: int, seed: int
) -> ExactBatch:
    batches = make_exact_batches(
        rows,
        physical_batch_size=physical_batch_size,
        captions_per_pair=captions_per_pair,
        epoch=0,
        seed=seed,
    )
    if not batches:
        raise ValueError(f"cannot form calibration batch of {physical_batch_size} pairs")
    return batches[0]


def run_candidate(
    args: argparse.Namespace,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    model_config: Siglip2TemporalConfig,
    physical_batch_size: int,
) -> dict[str, Any]:
    batch = candidate_batch(rows, physical_batch_size, args.captions_per_pair, args.seed)
    backbone = Siglip2Backbone(
        args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
    )
    backbone.enable_phase_b_top_blocks(
        2, gradient_checkpointing=model_config.gradient_checkpointing
    )
    model = Siglip2TemporalRetrievalModel(backbone, model_config).to(device)
    optimizer, optimizer_report, _scheduler = build_adamw(
        model, phase="B", total_steps=args.steps_per_candidate
    )
    image_inputs, text_inputs = processor_inputs(
        processor, batch.pair_rows, batch.query_rows, device
    )
    positive, ignored, _ = build_relevance_masks(
        batch.query_rows, batch.pair_rows, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    step_rows: list[dict[str, Any]] = []
    try:
        for step in range(1, args.steps_per_candidate + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**image_inputs, **text_inputs)
                loss = multi_positive_listwise_loss(
                    output.score_matrix.float(), positive, ignored
                )
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - backward_start
            gradients = module_gradient_report(model)
            flat = torch.cat(
                [
                    parameter.grad.detach().float().reshape(-1)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
            if not torch.isfinite(flat).all() or not torch.isfinite(loss):
                raise FloatingPointError("NONFINITE_CALIBRATION_STEP")
            grad_norm = float(flat.norm())
            optimizer_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_start
            step_rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm_preclip": grad_norm,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "score_shape": list(output.score_matrix.shape),
                    "gradient_report": gradients,
                }
            )
        result = {
            "status": "PASS",
            "physical_batch_size": physical_batch_size,
            "captions_per_pair": args.captions_per_pair,
            "query_count": len(batch.query_rows),
            "steps": args.steps_per_candidate,
            "score_matrix": list(output.score_matrix.shape),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "step_metrics": step_rows,
            "optimizer": optimizer_report,
        }
    except torch.cuda.OutOfMemoryError:
        result = {
            "status": "OOM",
            "physical_batch_size": physical_batch_size,
            "captions_per_pair": args.captions_per_pair,
            "steps_completed": len(step_rows),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "step_metrics": step_rows,
        }
    finally:
        del model, backbone, optimizer, image_inputs, text_inputs
        gc.collect()
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--steps-per-candidate", type=int, default=2)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        worktree = Path(__file__).resolve().parents[1]
        state = git_state(worktree)
        if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
            raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
        if args.steps_per_candidate <= 0 or args.steps_per_candidate * len(args.batch_sizes) > 32:
            raise ValueError("CALIBRATION_STEP_BUDGET_EXCEEDED")
        if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(
            value <= 0 for value in args.batch_sizes
        ):
            raise ValueError("batch sizes must be positive and unique")
        release = Path(args.data_release)
        manifest = Path(args.train_manifest)
        config_path = Path(args.config_path)
        for required in (release, manifest, config_path):
            if not required.exists():
                raise FileNotFoundError(required)
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("config must contain an object")
        config_fields = {
            name: config_payload[name]
            for name in Siglip2TemporalConfig.__dataclass_fields__
            if name in config_payload
        }
        model_config = Siglip2TemporalConfig.from_dict(config_fields)
        rows = load_exact_pair_rows(manifest, split="train")
        processor = AutoProcessor.from_pretrained(
            args.siglip2_model, local_files_only=True
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CALIBRATION_REQUIRES_CUDA")
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        results = []
        for batch_size in args.batch_sizes:
            results.append(
                run_candidate(
                    args,
                    processor,
                    rows,
                    device,
                    model_config,
                    batch_size,
                )
            )
        successful = [
            item
            for item in results
            if item["status"] == "PASS"
            and float(item["peak_reserved_gib"]) <= 68.0
        ]
        selected = max(
            (int(item["physical_batch_size"]) for item in successful), default=None
        )
        write_json(
            run / "calibration_results.json",
            {
                "status": "PASS" if selected is not None else "NO_ACCEPTED_BATCH",
                "results": results,
                "selected_physical_batch_size": selected,
                "selection_rule": "largest PASS with peak_reserved_gib <= 68",
                "logical_physical_batch_target": 128,
                "gradient_accumulation_if_selected": (
                    128 // selected if selected is not None else None
                ),
            },
        )
        write_json(
            run / "batch_contract.json",
            {
                "tested_physical_batches": args.batch_sizes,
                "steps_per_candidate": args.steps_per_candidate,
                "captions_per_pair": args.captions_per_pair,
                "logical_physical_batch_target": 128,
                "selected_physical_batch_size": selected,
                "selected_gradient_accumulation": (
                    128 // selected if selected is not None else None
                ),
            },
        )
        write_json(
            run / "cuda_memory.json",
            {
                "gpu": torch.cuda.get_device_name(device),
                "calibration_peak_memory_is_per_candidate": True,
                "cpu_peak_rss_gib": peak_cpu_rss_gib(),
            },
        )
        write_json(
            run / "code_state.json",
            {"expected_code_sha": args.expected_code_sha, "runtime_git": state},
        )
        write_json(
            run / "data_contract.json",
            {
                "data_release": str(release),
                "data_release_sha256": sha256_path(release),
                "train_manifest": str(manifest),
                "train_manifest_sha256": sha256(manifest),
                "allowed_sources": ["levir_mci", "second_cc"],
                "mask_access": False,
                "generated_unverified_text": False,
            },
        )
        write_json(
            run / "model_source.json",
            {
                "repository": "google/siglip2-base-patch16-256",
                "local_path": args.siglip2_model,
                "weights_sha256": sha256(Path(args.siglip2_model) / "model.safetensors"),
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "model_config": model_config.to_dict(),
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
            run / "training_complete.json",
            {
                "status": "PASS" if selected is not None else "NO_ACCEPTED_BATCH",
                "steps_total": args.steps_per_candidate * len(args.batch_sizes),
                "selected_physical_batch_size": selected,
                "no_nan_or_oom_for_selected": selected is not None,
            },
        )
        write_sha256sums(run)
        return 0 if selected is not None else 2
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
