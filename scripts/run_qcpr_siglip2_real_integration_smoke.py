#!/usr/bin/env python3
"""Run one bounded real-image SigLIP-2 integration smoke.

This is an integration contract, not a scientific training run.  It uses the
real Dataset-v2 exact-core images and captions, the pinned local SigLIP-2
checkpoint and the real tokenizer.  The hard limit is 32 optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
from run_qcpr_siglip2_phase import (
    checkpoint_roundtrip,
    git_state,
    sha256,
    sha256_path,
    write_json,
    write_sha256sums,
)
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.loader import ExactBatch, make_exact_batches
from qcpr_siglip2.data.manifest import load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import (
    _device_autocast,
    build_relevance_masks,
    processor_image_inputs,
    processor_text_inputs,
)
from qcpr_siglip2.evaluation.evidence import (
    query_swap_map_cosine,
    query_swap_map_l1,
    time_reversal_score_change,
)
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.exposure import ExposureLedger
from qcpr_siglip2.training.gradcache import module_gradient_report
from qcpr_siglip2.training.objective import multi_positive_listwise_loss
from qcpr_siglip2.training.optimizer import build_adamw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    # The value is still passed explicitly to the processor.  It is not read
    # from the processor's internal default, which is the prohibited silent
    # 256-token cap.
    parser.add_argument("--max-num-patches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def _write_failure(run: Path, exc: Exception) -> None:
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


def _load_config(path: Path) -> Siglip2TemporalConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config must contain a JSON object")
    fields = {
        name: payload[name]
        for name in Siglip2TemporalConfig.__dataclass_fields__
        if name in payload
    }
    return Siglip2TemporalConfig.from_dict(fields)


def _reverse_image_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reversed_inputs = dict(inputs)
    for key, value in inputs.items():
        if (
            (key == "pixel_values" or key in {"pixel_attention_mask", "spatial_shapes"})
            and value.ndim >= 2
        ):
            reversed_inputs[key] = value.flip(1)
    return reversed_inputs


def _score_from_weights(
    model: Siglip2TemporalRetrievalModel,
    output: Any,
    query_index: int,
    pair_index: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    visual = output.temporal.temporal_patch_tokens[pair_index]
    vector = torch.einsum("m,md->d", weights, visual)
    query_vector = vector.view(1, 1, -1).expand(
        output.text_embedding.shape[0], 1, -1
    )
    token_evidence_score = output.evidence.evidence_score.clone()
    token_evidence_score[query_index, pair_index] = (
        weights * output.evidence.evidence_logits[query_index, pair_index]
    ).sum()
    score = model.unified_score_from_evidence(
        output.text_embedding,
        output.pair_cls,
        output.temporal.change_tokens,
        output.evidence.evidence_gate,
        query_vector,
        token_evidence_score=token_evidence_score,
    )
    return score[query_index, pair_index]


def _evidence_deletion(
    model: Siglip2TemporalRetrievalModel, output: Any, query_index: int, pair_index: int
) -> dict[str, float | bool]:
    weights = output.evidence.evidence_weights[query_index, pair_index].detach()
    token_count = int(weights.numel())
    count = max(1, int(token_count * 0.10))
    descending = weights.argsort(descending=True)
    ascending = weights.argsort(descending=False)
    top_mask = torch.ones_like(weights, dtype=torch.bool)
    bottom_mask = torch.ones_like(weights, dtype=torch.bool)
    top_mask[descending[:count]] = False
    bottom_mask[ascending[:count]] = False
    top_weights = weights * top_mask
    bottom_weights = weights * bottom_mask
    top_weights = top_weights / top_weights.sum().clamp_min(1e-12)
    bottom_weights = bottom_weights / bottom_weights.sum().clamp_min(1e-12)
    normal = output.score_matrix[query_index, pair_index].detach()
    top_score = _score_from_weights(model, output, query_index, pair_index, top_weights)
    bottom_score = _score_from_weights(
        model, output, query_index, pair_index, bottom_weights
    )
    normal = normal.float()
    top_score = top_score.float()
    bottom_score = bottom_score.float()
    top_drop = float((normal - top_score).detach().cpu())
    bottom_drop = float((normal - bottom_score).detach().cpu())
    return {
        "top_fraction": 0.10,
        "top_score_drop": top_drop,
        "bottom_score_drop": bottom_drop,
        "passed": top_drop > bottom_drop,
    }


def _detach_diagnostic(
    model: Siglip2TemporalRetrievalModel,
    image_inputs: dict[str, torch.Tensor],
    text_inputs: dict[str, torch.Tensor],
) -> dict[str, float | bool]:
    model.train()
    with _device_autocast(torch.device("cuda"), torch.bfloat16):
        output = model(**image_inputs, **text_inputs)
    output.evidence.evidence_vector.retain_grad()
    output.score_matrix[0, 0].backward()
    gradient = output.evidence.evidence_vector.grad
    gradient_norm = float(gradient.detach().float().norm()) if gradient is not None else 0.0
    model.zero_grad(set_to_none=True)

    with _device_autocast(torch.device("cuda"), torch.bfloat16):
        detached_output = model(**image_inputs, **text_inputs)
    detached_output.evidence.evidence_vector.retain_grad()
    detached_vector = detached_output.evidence.evidence_vector.detach()
    with _device_autocast(torch.device("cuda"), torch.bfloat16):
        detached_score = model.unified_score_from_evidence(
            detached_output.text_embedding,
            detached_output.pair_cls,
            detached_output.temporal.change_tokens,
            detached_output.evidence.evidence_gate,
            detached_vector,
        )
    detached_score[0, 0].backward()
    detached_gradient = detached_output.evidence.evidence_vector.grad
    model.zero_grad(set_to_none=True)
    return {
        "normal_evidence_gradient_norm": gradient_norm,
        "detached_evidence_gradient_is_none": detached_gradient is None,
        "passed": gradient_norm > 0.0 and detached_gradient is None,
    }


def _query_and_temporal_diagnostics(
    model: Siglip2TemporalRetrievalModel,
    output: Any,
    image_inputs: dict[str, torch.Tensor],
    text_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    pair_index = 0
    query_a, query_b = 0, 1
    first_map = output.evidence.evidence_map[query_a, pair_index]
    second_map = output.evidence.evidence_map[query_b, pair_index]
    normal_score = output.score_matrix[query_a, pair_index]
    zero_weights = torch.zeros_like(output.evidence.evidence_weights[query_a, pair_index])
    zero_score = _score_from_weights(model, output, query_a, pair_index, zero_weights)
    with torch.no_grad(), _device_autocast(torch.device("cuda"), torch.bfloat16):
        reversed_output = model(
            **_reverse_image_inputs(image_inputs), **text_inputs
        )
    deletion = _evidence_deletion(model, output, query_a, pair_index)
    detach = _detach_diagnostic(model, image_inputs, text_inputs)
    diagnostics = {
        "query_swap_map_l1": query_swap_map_l1(first_map, second_map),
        "query_swap_map_cosine": query_swap_map_cosine(first_map, second_map),
        "score_change_after_query_swap": float(
            (
                output.score_matrix[query_a, pair_index].float()
                - output.score_matrix[query_b, pair_index].float()
            )
            .abs()
            .detach()
            .cpu()
        ),
        "score_with_zero_evidence": float(zero_score.detach().cpu()),
        "normal_score": float(normal_score.detach().cpu()),
        "score_change_after_evidence_zeroing": float(
            (normal_score.float() - zero_score.float()).abs().detach().cpu()
        ),
        "evidence_zeroing_passed": bool(
            not torch.allclose(
                normal_score.float(), zero_score.float(), atol=1e-7, rtol=1e-6
            )
        ),
        "deletion": deletion,
        "detach": detach,
        "time_reversal_score_change": time_reversal_score_change(
            output.score_matrix, reversed_output.score_matrix
        ),
        "query_conditioned_passed": bool(
            query_swap_map_l1(first_map, second_map) > 1e-7
            and abs(
                float(
                    output.score_matrix[query_a, pair_index].float()
                    - output.score_matrix[query_b, pair_index].float()
                )
            )
            > 1e-7
        ),
        "temporal_direction_passed": bool(
            time_reversal_score_change(
                output.score_matrix, reversed_output.score_matrix
            ) > 1e-7
        ),
    }
    diagnostics["passed"] = bool(
        diagnostics["query_conditioned_passed"]
        and diagnostics["evidence_zeroing_passed"]
        and deletion["passed"]
        and detach["passed"]
        and diagnostics["temporal_direction_passed"]
    )
    return diagnostics


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        if not 1 <= args.steps <= 32:
            raise ValueError("REAL_SMOKE_STEP_LIMIT_EXCEEDED")
        if args.physical_batch_size <= 0 or args.captions_per_pair <= 0:
            raise ValueError("invalid batch contract")
        worktree = Path(__file__).resolve().parents[1]
        state = git_state(worktree)
        if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
            raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
        data_release = Path(args.data_release)
        train_manifest = Path(args.train_manifest)
        development_manifest = Path(args.development_manifest)
        config_path = Path(args.config_path)
        for required in (data_release, train_manifest, development_manifest, config_path):
            if not required.exists():
                raise FileNotFoundError(required)
        if not Path(args.siglip2_model).is_dir():
            raise FileNotFoundError(args.siglip2_model)

        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("REAL_INTEGRATION_SMOKE_REQUIRES_CUDA")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

        train_rows = load_exact_pair_rows(train_manifest, split="train")
        development_rows = load_exact_pair_rows(
            development_manifest, split="development"
        )
        batches = make_exact_batches(
            train_rows,
            physical_batch_size=args.physical_batch_size,
            captions_per_pair=args.captions_per_pair,
            epoch=0,
            seed=args.seed,
        )
        if not batches:
            raise ValueError("cannot form the real integration smoke batch")
        batch: ExactBatch = batches[0]
        positive, ignored, relevance = build_relevance_masks(
            batch.query_rows, batch.pair_rows, device
        )
        processor = AutoProcessor.from_pretrained(
            args.siglip2_model, local_files_only=True
        )
        config = _load_config(config_path)
        backbone = Siglip2Backbone(
            args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
        )
        model = Siglip2TemporalRetrievalModel(backbone, config).to(device)
        optimizer, optimizer_report, scheduler = build_adamw(
            model, phase="A", total_steps=args.steps
        )
        write_json(run / "parameter_groups.json", optimizer_report)
        write_json(
            run / "code_state.json",
            {"expected_code_sha": args.expected_code_sha, "runtime_git": state},
        )
        write_json(
            run / "model_source.json",
            {
                "repository": config.backbone,
                "local_path": args.siglip2_model,
                "weights_sha256": sha256(Path(args.siglip2_model) / "model.safetensors"),
                "runtime_class": backbone.runtime_class,
                "is_naflex": backbone.is_naflex,
                "max_num_patches": args.max_num_patches,
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
            run / "data_contract.json",
            {
                "release": str(data_release),
                "release_sha256": sha256_path(data_release),
                "train_manifest": str(train_manifest),
                "train_manifest_sha256": sha256(train_manifest),
                "development_manifest": str(development_manifest),
                "development_manifest_sha256": sha256(development_manifest),
                "train_ordered_pair_sha256": ordered_id_sha256(
                    train_rows, "canonical_pair_id"
                ),
                "development_ordered_query_sha256": ordered_id_sha256(
                    development_rows, "caption_id"
                ),
                "allowed_sources": ["levir_mci", "second_cc"],
                "mask_access": False,
                "generated_unverified_text": False,
            },
        )
        write_json(
            run / "batch_contract.json",
            {
                "steps": args.steps,
                "physical_batch_size": args.physical_batch_size,
                "captions_per_pair": args.captions_per_pair,
                "query_count": len(batch.query_rows),
                "score_matrix": [len(batch.query_rows), len(batch.pair_rows)],
                "multi_positive_queries": relevance["multi_positive_queries"],
                "multi_positive_runtime_status": (
                    "EXERCISED"
                    if relevance["multi_positive_queries"]
                    else "MULTI_POSITIVE_RUNTIME_NOT_EXERCISED"
                ),
                "precision": "bf16",
                "mask_access": False,
            },
        )

        ledger = ExposureLedger()
        metric_rows: list[dict[str, Any]] = []
        timing = {
            "image_decode_seconds": 0.0,
            "text_tokenization_seconds": 0.0,
            "forward_seconds": 0.0,
            "backward_seconds": 0.0,
            "optimizer_seconds": 0.0,
        }
        total_start = time.perf_counter()
        last_output: Any = None
        last_image_inputs: dict[str, torch.Tensor] | None = None
        last_text_inputs: dict[str, torch.Tensor] | None = None
        final_gradient_report: dict[str, Any] = {}
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            image_start = time.perf_counter()
            image_inputs = processor_image_inputs(
                processor,
                batch.pair_rows,
                device,
                max_num_patches=args.max_num_patches,
                is_naflex=backbone.is_naflex,
            )
            timing["image_decode_seconds"] += time.perf_counter() - image_start
            text_start = time.perf_counter()
            text_inputs = processor_text_inputs(processor, batch.query_rows, device)
            timing["text_tokenization_seconds"] += time.perf_counter() - text_start
            torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            with _device_autocast(device, torch.bfloat16):
                output = model(**image_inputs, **text_inputs)
                loss = multi_positive_listwise_loss(
                    output.score_matrix.float(), positive, ignored
                )
            torch.cuda.synchronize(device)
            timing["forward_seconds"] += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            timing["backward_seconds"] += time.perf_counter() - backward_start
            final_gradient_report = module_gradient_report(model)
            gradients = [
                parameter.grad.detach().float().reshape(-1)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            flat = torch.cat(gradients) if gradients else torch.empty(0, device=device)
            if not torch.isfinite(loss) or not torch.isfinite(flat).all():
                raise FloatingPointError("NONFINITE_GRADIENT")
            optimizer_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            timing["optimizer_seconds"] += time.perf_counter() - optimizer_start
            ledger.record_step(batch.pair_ids, batch.query_ids)
            last_output = output
            last_image_inputs = image_inputs
            last_text_inputs = text_inputs
            metric_rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "score_shape": list(output.score_matrix.shape),
                    "gradient_norm_preclip": float(flat.norm().cpu()),
                    "gradient_report": final_gradient_report,
                    "finite": True,
                }
            )
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
                encoding="utf-8",
            )
        total_wall = time.perf_counter() - total_start
        if len(metric_rows) != args.steps:
            raise RuntimeError("FIXED_STEP_CONTRACT_NOT_SATISFIED")

        assert last_output is not None
        assert last_image_inputs is not None and last_text_inputs is not None
        evidence_diagnostics = _query_and_temporal_diagnostics(
            model, last_output, last_image_inputs, last_text_inputs
        )
        write_json(run / "evidence_diagnostics.json", evidence_diagnostics)
        if not evidence_diagnostics["passed"]:
            raise RuntimeError("EVIDENCE_MECHANISM_GATE_FAILED")

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": args.steps,
            "metadata": {
                "code_sha": args.expected_code_sha,
                "data_release": str(data_release),
                "train_manifest": str(train_manifest),
                "fixed_steps": args.steps,
                "optimizer_resumed": False,
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }
        checkpoint = run / "checkpoint.pt"
        checkpoint_start = time.perf_counter()
        torch.save(checkpoint_payload, checkpoint)
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        checkpoint_hash = sha256(checkpoint)
        (run / "checkpoint.sha256").write_text(
            f"{checkpoint_hash}  {checkpoint.name}\n", encoding="utf-8"
        )
        roundtrip = checkpoint_roundtrip(
            model,
            backbone,
            processor,
            checkpoint,
            checkpoint_payload,
            config,
            args.siglip2_model,
            "A",
            batch,
            device,
            args.max_num_patches,
        )
        write_json(run / "checkpoint_roundtrip.json", roundtrip)
        write_json(run / "exposure_accounting.json", ledger.to_dict())
        write_json(
            run / "gradient_diagnostics.json",
            {
                "status": "PASS",
                "last_gradient_norm_preclip": metric_rows[-1]["gradient_norm_preclip"],
                "last_module_gradient_report": final_gradient_report,
                "all_losses_finite": True,
                "all_gradients_finite": True,
            },
        )
        write_json(
            run / "runtime_profile.json",
            {
                "total_wall_seconds": total_wall,
                "seconds_per_step": total_wall / args.steps,
                "image_decoding_seconds": timing["image_decode_seconds"],
                "text_tokenization_seconds": timing["text_tokenization_seconds"],
                "forward_seconds": timing["forward_seconds"],
                "backward_seconds": timing["backward_seconds"],
                "optimizer_seconds": timing["optimizer_seconds"],
                "checkpoint_write_seconds": checkpoint_seconds,
                "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
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
            run / "model_contract.json",
            {
                "runtime_class": backbone.runtime_class,
                "backbone_parameter_scope": backbone.parameter_scope_report(),
                "native_visual_token_shape": [
                    len(batch.pair_rows),
                    last_output.temporal.frame_count,
                    last_output.temporal.patch_count,
                    config.hidden_size,
                ],
                "temporal_patch_token_shape": list(
                    last_output.temporal.temporal_patch_tokens.shape
                ),
                "text_token_shape": [
                    len(batch.query_rows),
                    int(last_text_inputs["input_ids"].shape[-1]),
                    config.hidden_size,
                ],
                "score_matrix_shape": list(last_output.score_matrix.shape),
                "hidden_size": config.hidden_size,
                "expected_patch_tokens": config.expected_patch_tokens,
                "mask_access": False,
            },
        )
        not_run = {
            "status": "NOT_RUN",
            "reason": "bounded integration smoke does not substitute for common-gallery evaluation",
        }
        write_json(run / "evaluation_metrics.json", not_run)
        torch.save(not_run, run / "full_rankings.pt")
        (run / "rankings_top100.jsonl").write_text(
            json.dumps(not_run) + "\n", encoding="utf-8"
        )
        write_json(
            run / "training_complete.json",
            {
                "status": "PASS",
                "global_step": args.steps,
                "requested_steps": args.steps,
                "no_nan_or_oom": True,
                "checkpoint": str(checkpoint),
                "evidence_gate": "PASS",
                "evaluation_status": "NOT_RUN",
            },
        )
        write_sha256sums(run)
        return 0
    except Exception as exc:
        _write_failure(run, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
