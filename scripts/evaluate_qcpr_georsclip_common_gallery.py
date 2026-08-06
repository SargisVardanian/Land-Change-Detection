#!/usr/bin/env python3
"""Run the frozen GeoRSCLIP step-zero common-gallery evaluation.

This is a separate diagnostic from the SigLIP-2 track.  The GeoRSCLIP
vision/text towers are frozen and the temporal head is freshly initialized;
no optimizer step is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from torch import Tensor

from qcpr_siglip2.backbones.georsclip import GeoRSCLIPBackbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.manifest import load_exact_core_rows, ordered_id_sha256
from qcpr_siglip2.evaluation.common_gallery import (
    audit_ranking_integrity,
    canonical_pair_rows,
    exact_relevance_masks,
    global_stage_scores,
    merge_reranked_scores,
    ranking_records,
    write_jsonl,
)
from qcpr_siglip2.evaluation.retrieval import full_gallery_metrics
from qcpr_siglip2.evaluation.reranking import select_topk_candidates
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel

FORBIDDEN_MASK_KEYS = frozenset(
    {
        "mask",
        "mask_path",
        "dense_label_path",
        "official_label",
        "semantic_map",
        "semantic_map_path",
        "binary_change_path",
        "change_mask_path",
        "label_path",
    }
)


def _mask_violations(value: Any, path: str = "row") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            name = str(key)
            if name in FORBIDDEN_MASK_KEYS:
                violations.append(f"{path}.{name}")
            violations.extend(_mask_violations(nested, f"{path}.{name}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(_mask_violations(nested, f"{path}[{index}]"))
    return violations


def assert_mask_free(rows: list[dict[str, Any]]) -> None:
    violations: list[str] = []
    for index, row in enumerate(rows):
        violations.extend(_mask_violations(row, f"row[{index}]"))
    if violations:
        raise RuntimeError("MASK_FREE_MANIFEST_VIOLATION:" + ",".join(violations[:10]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sha256sums(run: Path) -> None:
    lines = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{_sha256_file(path)}  {path.name}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_state(worktree: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"head": head, "worktree_clean": not dirty}


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def _preprocess_and_stack(preprocess: Any, rows: list[dict[str, Any]], device: torch.device) -> Tensor:
    images: list[Tensor] = []
    for row in rows:
        for field in ("t1_path", "t2_path"):
            path = Path(str(row[field]))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
    return torch.stack(images, dim=0).reshape(
        len(rows), 2, *images[0].shape
    ).to(device)


@torch.no_grad()
def _encode_gallery(
    backbone: GeoRSCLIPBackbone,
    preprocess: Any,
    model: Siglip2TemporalRetrievalModel,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    frame_tokens: list[Tensor] = []
    frame_embeddings: list[Tensor] = []
    pair_embeddings: list[Tensor] = []
    for start in range(0, len(rows), batch_size):
        pixel_values = _preprocess_and_stack(preprocess, rows[start : start + batch_size], device)
        with _autocast(device):
            image = backbone.encode_images(pixel_values)
            temporal = model.temporal_adapter(image.patch_tokens, image.pooled_embedding)
        frame_tokens.append(image.patch_tokens.detach().cpu())
        frame_embeddings.append(image.pooled_embedding.detach().cpu())
        pair_embeddings.append(torch.nn.functional.normalize(temporal.pair_cls.float(), dim=-1).cpu())
    return (
        torch.cat(frame_tokens),
        torch.cat(frame_embeddings),
        torch.cat(pair_embeddings),
    )


@torch.no_grad()
def _encode_queries(
    backbone: GeoRSCLIPBackbone,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    token_embeddings: list[Tensor] = []
    pooled_embeddings: list[Tensor] = []
    attention_masks: list[Tensor] = []
    for start in range(0, len(rows), batch_size):
        input_ids = tokenizer([str(row["caption"]) for row in rows[start : start + batch_size]]).to(device)
        attention_mask = input_ids.ne(0)
        with _autocast(device):
            text = backbone.encode_text(input_ids, attention_mask)
        token_embeddings.append(text.token_embeddings.detach().cpu())
        pooled_embeddings.append(text.pooled_embedding.detach().cpu())
        attention_masks.append(text.attention_mask.detach().cpu())
    return (
        torch.cat(token_embeddings),
        torch.cat(pooled_embeddings),
        torch.cat(attention_masks),
    )


@torch.no_grad()
def _rerank(
    model: Siglip2TemporalRetrievalModel,
    frame_tokens: Tensor,
    frame_embeddings: Tensor,
    text_tokens: Tensor,
    text_embeddings: Tensor,
    text_mask: Tensor,
    global_scores: Tensor,
    device: torch.device,
    max_k: int,
    query_batch_size: int,
) -> tuple[Tensor, Tensor]:
    candidates = select_topk_candidates(global_scores, max_k)
    reranked = global_scores.clone()
    maps: list[Tensor] = []
    model.eval()
    for start in range(0, text_tokens.shape[0], query_batch_size):
        end = min(start + query_batch_size, text_tokens.shape[0])
        candidate_block = candidates[start:end]
        unique = sorted({int(value) for value in candidate_block.reshape(-1).tolist()})
        local = {value: index for index, value in enumerate(unique)}
        local_indices = torch.tensor(unique, dtype=torch.long)
        with _autocast(device):
            output = model.forward_from_features(
                frame_tokens[local_indices].to(device),
                frame_embeddings[local_indices].to(device),
                text_tokens[start:end].to(device),
                text_embeddings[start:end].to(device),
                text_mask[start:end].to(device),
            )
        local_candidates = torch.tensor(
            [[local[int(value)] for value in row] for row in candidate_block.tolist()],
            dtype=torch.long,
        )
        values = output.score_matrix.float().cpu().gather(1, local_candidates)
        reranked = merge_reranked_scores(reranked, candidate_block, values)
        top1_local = local[int(candidate_block[0, 0])]
        maps.append(output.evidence.evidence_map[:, top1_local].float().cpu())
    return reranked, torch.cat(maps)


def _effective_rank(values: Tensor) -> float:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def _summary(values: Tensor) -> dict[str, float]:
    flat = values.float().reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99]))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--georsclip-checkpoint", required=True)
    parser.add_argument("--georsclip-revision", required=True)
    parser.add_argument(
        "--temporal-checkpoint",
        help="Optional trained temporal-head checkpoint. Omit for frozen step-0.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gallery-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--rerank-query-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        worktree = Path(args.worktree)
        state = _git_state(worktree)
        if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
            raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
        release = Path(args.data_release)
        manifest = Path(args.development_manifest)
        checkpoint = Path(args.georsclip_checkpoint)
        for required in (release, manifest, checkpoint):
            if not required.exists():
                raise FileNotFoundError(required)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EVALUATION_REQUIRES_CUDA")
        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        core_rows = load_exact_core_rows(manifest, split="development")
        query_rows = [row for row in core_rows if row.get("query_scope") == "exact_pair"]
        pair_rows = canonical_pair_rows(core_rows)
        assert_mask_free(core_rows)
        positive, ignored = exact_relevance_masks(query_rows, pair_rows)

        import open_clip

        preprocess = open_clip.image_transform(
            (224, 224),
            is_train=False,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            interpolation="bicubic",
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        backbone = GeoRSCLIPBackbone(checkpoint, model_name="ViT-B-32").to(device)
        config = Siglip2TemporalConfig(
            hidden_size=512,
            expected_patch_tokens=49,
            base_grid=7,
            attention_heads=8,
            mlp_size=2048,
            max_frames=2,
        )
        model = Siglip2TemporalRetrievalModel(None, config).to(device)
        temporal_checkpoint_sha = None
        temporal_global_step = 0
        if args.temporal_checkpoint is not None:
            temporal_checkpoint = Path(args.temporal_checkpoint)
            if not temporal_checkpoint.is_file():
                raise FileNotFoundError(temporal_checkpoint)
            payload = torch.load(
                temporal_checkpoint, map_location="cpu", weights_only=True
            )
            if not isinstance(payload, dict) or "model_state" not in payload:
                raise ValueError("temporal checkpoint lacks model_state")
            checkpoint_config = payload.get("config")
            if checkpoint_config is not None:
                if checkpoint_config != config.to_dict():
                    raise ValueError("temporal checkpoint config mismatch")
            model.load_state_dict(payload["model_state"], strict=True)
            temporal_checkpoint_sha = _sha256_file(temporal_checkpoint)
            temporal_global_step = int(payload.get("global_step", 0))
        model.eval()
        started = time.perf_counter()
        frame_tokens, frame_embeddings, pair_embeddings = _encode_gallery(
            backbone, preprocess, model, pair_rows, device, args.gallery_batch_size
        )
        text_tokens, text_embeddings, text_mask = _encode_queries(
            backbone, tokenizer, query_rows, device, args.query_batch_size
        )
        text_normalized = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
        temperature = float(model.retrieval_temperature.detach().cpu())
        global_scores = global_stage_scores(text_normalized, pair_embeddings, temperature)
        global_scores = global_scores.masked_fill(ignored, -1.0e4)
        global_metrics = full_gallery_metrics(global_scores, positive)
        max_k = min(100, len(pair_rows))
        reranked_scores, evidence_maps = _rerank(
            model,
            frame_tokens,
            frame_embeddings,
            text_tokens,
            text_embeddings,
            text_mask,
            global_scores,
            device,
            max_k,
            args.rerank_query_batch_size,
        )
        reranked = {}
        for requested_k in (20, 50, 100):
            candidates = select_topk_candidates(global_scores, requested_k)
            values = reranked_scores.gather(1, candidates)
            scores = merge_reranked_scores(global_scores, candidates, values)
            reranked[str(requested_k)] = full_gallery_metrics(scores, positive)

        integrity = audit_ranking_integrity(global_scores, query_rows, pair_rows)
        integrity.update(
            {
                "development_manifest_sha256": _sha256_file(manifest),
                "data_release_sha256": _sha256_path(release),
                "ordered_query_ids_sha256": ordered_id_sha256(query_rows, "caption_id"),
                "ordered_gallery_ids_sha256": hashlib.sha256(
                    ("\n".join(str(row["canonical_pair_id"]) for row in pair_rows) + "\n").encode()
                ).hexdigest(),
                "mask_free": True,
            }
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "seed": args.seed,
            },
            run / "checkpoint.pt",
        )
        torch.save(
            {
                "global_scores": global_scores,
                "positive_mask": positive,
                "ignored_mask": ignored,
                "reranked_scores": reranked_scores,
            },
            run / "full_rankings.pt",
        )
        write_jsonl(
            run / "rankings_top100.jsonl",
            ranking_records(reranked_scores, query_rows, pair_rows, top_k=100),
        )
        torch.save(evidence_maps, run / "evidence_maps_top1.pt")
        embeddings = {
            "pair_effective_rank": _effective_rank(pair_embeddings),
            "text_effective_rank": _effective_rank(text_embeddings),
            "pair_norms": _summary(pair_embeddings.norm(dim=-1)),
            "text_norms": _summary(text_embeddings.norm(dim=-1)),
            "positive_scores": _summary(global_scores[positive]),
            "negative_scores": _summary(global_scores[~positive]),
        }
        map_flat = evidence_maps.reshape(evidence_maps.shape[0], -1).float()
        probabilities = map_flat / map_flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        evidence = {
            "entropy": _summary(entropy),
            "effective_token_count": _summary(entropy.exp()),
            "nearly_uniform_fraction": float(
                (entropy > 0.95 * torch.log(torch.tensor(float(map_flat.shape[1])))).float().mean()
            ),
            "single_token_fraction": float((probabilities.max(dim=1).values > 0.95).float().mean()),
            "map_shape": list(evidence_maps.shape),
        }
        checkpoint_sha = _sha256_file(run / "checkpoint.pt")
        _write_json(
            run / "model_source.json",
            {
                "repository": "Zilun/GeoRSCLIP",
                "revision": args.georsclip_revision,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "model_name": "ViT-B-32",
                "weights_only": True,
                "license_status": "RESEARCH_BASELINE_ONLY",
                "load_audit": backbone.parameter_scope_report()["load_audit"],
            },
        )
        _write_json(run / "model_contract.json", {
            "runtime_class": type(backbone.model).__name__,
            "native_visual_tokens": list(frame_tokens.shape),
            "text_tokens": list(text_tokens.shape),
            "pair_embedding_shape": list(pair_embeddings.shape),
            "score_matrix_shape": list(global_scores.shape),
            "hidden_size": 512,
            "patch_grid": 7,
            "mask_access": False,
            "temperature": temperature,
            "temporal_checkpoint": args.temporal_checkpoint,
            "temporal_checkpoint_sha256": temporal_checkpoint_sha,
            "temporal_global_step": temporal_global_step,
        })
        _write_json(run / "batch_contract.json", {
            "query_count": len(query_rows),
            "gallery_count": len(pair_rows),
            "score_matrix": list(global_scores.shape),
            "gallery_batch_size": args.gallery_batch_size,
            "query_batch_size": args.query_batch_size,
            "rerank_query_batch_size": args.rerank_query_batch_size,
            "optimizer_steps": 0,
            "mask_access": False,
            "temporal_checkpoint": args.temporal_checkpoint,
            "temporal_global_step": temporal_global_step,
        })
        _write_json(run / "training_complete.json", {
            "status": (
                "TRAINED_TEMPORAL_HEAD_EVALUATION_PASS"
                if args.temporal_checkpoint is not None
                else "FROZEN_EVALUATION_PASS"
            ),
            "global_step": temporal_global_step,
            "optimizer_steps": 0,
            "checkpoint_sha256": checkpoint_sha,
            "scientific_interpretation": (
                "GeoRSCLIP frozen towers with an externally trained temporal head"
                if args.temporal_checkpoint is not None
                else "frozen GeoRSCLIP step-zero diagnostic; no training"
            ),
        })
        _write_json(run / "evaluation_metrics.json", {
            "protocol": "QCPR_GEORSCLIP_FROZEN_STEP0_COMMON_GALLERY",
            "query_count": len(query_rows),
            "gallery_count": len(pair_rows),
            "global": global_metrics,
            "reranked": reranked,
        })
        _write_json(run / "ranking_integrity.json", integrity)
        _write_json(run / "embedding_diagnostics.json", embeddings)
        _write_json(run / "evidence_diagnostics.json", evidence)
        _write_json(run / "parameter_groups.json", {
            "optimizer_steps": 0,
            "georsclip_vision_frozen": True,
            "georsclip_text_frozen": True,
            "temporal_head_initialized_only": True,
            "trainable_parameter_count": 0,
        })
        _write_json(run / "runtime_profile.json", {
            "total_wall_seconds": time.perf_counter() - started,
            "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        })
        _write_json(run / "cuda_memory.json", {
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else None,
        })
        _write_sha256sums(run)
        return 0
    except Exception as exc:
        _write_json(run / "failure.json", {
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "expected_code_sha": args.expected_code_sha,
        })
        _write_sha256sums(run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
