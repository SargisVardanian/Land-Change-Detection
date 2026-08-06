#!/usr/bin/env python3
"""Evaluate the SigLIP-2 track on one immutable exact development gallery.

The script keeps stage-1 global retrieval and stage-2 Top-K evidence
reranking separate.  It never opens dense labels or mask sidecars.
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
from torch import Tensor
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.manifest import load_exact_pair_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import encode_real_images, encode_real_text
from qcpr_siglip2.evaluation.common_gallery import (
    audit_ranking_integrity,
    canonical_pair_rows,
    exact_relevance_masks,
    global_stage_scores,
    merge_reranked_scores,
    metrics_by_query_group,
    ranking_records,
    write_jsonl,
)
from qcpr_siglip2.evaluation.reranking import select_topk_candidates
from qcpr_siglip2.evaluation.retrieval import full_gallery_metrics
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory deterministically without absolute paths."""

    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sha256sums(run: Path) -> None:
    lines = []
    for path in sorted(run.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{_sha256_file(path)}  {path.name}")
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


def _contains_forbidden(value: Any, path: str = "row") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_string = str(key)
            if key_string in FORBIDDEN_MASK_KEYS:
                violations.append(f"{path}.{key_string}")
            violations.extend(_contains_forbidden(nested, f"{path}.{key_string}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(_contains_forbidden(nested, f"{path}[{index}]"))
    return violations


def assert_mask_free(rows: list[dict[str, Any]]) -> None:
    violations = []
    for index, row in enumerate(rows):
        violations.extend(_contains_forbidden(row, f"row[{index}]"))
    if violations:
        raise RuntimeError("MASK_FREE_MANIFEST_VIOLATION:" + ",".join(violations[:10]))


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def encode_gallery_features(
    model: Siglip2TemporalRetrievalModel,
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    frame_tokens: list[torch.Tensor] = []
    frame_embeddings: list[torch.Tensor] = []
    pair_embeddings: list[torch.Tensor] = []
    for start in range(0, len(pair_rows), batch_size):
        image = encode_real_images(
            backbone, processor, pair_rows[start : start + batch_size], device
        )
        with torch.no_grad(), _autocast(device):
            temporal = model.temporal_adapter(
                image.patch_tokens, image.pooled_embedding
            )
        frame_tokens.append(image.patch_tokens.detach().to("cpu"))
        frame_embeddings.append(image.pooled_embedding.detach().to("cpu"))
        pair_embeddings.append(
            torch.nn.functional.normalize(temporal.pair_cls.float(), dim=-1).cpu()
        )
    return (
        torch.cat(frame_tokens, dim=0),
        torch.cat(frame_embeddings, dim=0),
        torch.cat(pair_embeddings, dim=0),
    )


def encode_queries(
    backbone: Siglip2Backbone,
    processor: Any,
    query_rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    token_embeddings: list[torch.Tensor] = []
    pooled_embeddings: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    for start in range(0, len(query_rows), batch_size):
        text = encode_real_text(
            backbone, processor, query_rows[start : start + batch_size], device
        )
        token_embeddings.append(text.token_embeddings.detach().to("cpu"))
        pooled_embeddings.append(text.pooled_embedding.detach().to("cpu"))
        attention_masks.append(text.attention_mask.detach().to("cpu"))
    return (
        torch.cat(token_embeddings, dim=0),
        torch.cat(pooled_embeddings, dim=0),
        torch.cat(attention_masks, dim=0),
    )


def rerank_top_k(
    model: Siglip2TemporalRetrievalModel,
    gallery_frame_tokens: Tensor,
    gallery_frame_embeddings: Tensor,
    query_token_embeddings: Tensor,
    query_pooled_embeddings: Tensor,
    query_attention_masks: Tensor,
    global_scores: torch.Tensor,
    device: torch.device,
    *,
    max_k: int,
    query_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rerank the global Top-max_k and return scores plus Top-1 maps."""

    candidates = select_topk_candidates(global_scores, max_k)
    reranked = global_scores.clone()
    top1_maps: list[torch.Tensor] = []
    model.eval()
    query_count = int(query_token_embeddings.shape[0])
    for start in range(0, query_count, query_batch_size):
        end = min(start + query_batch_size, query_count)
        candidate_block = candidates[start:end]
        unique_indices = sorted(
            {int(value) for value in candidate_block.reshape(-1).tolist()}
        )
        local_index = {value: index for index, value in enumerate(unique_indices)}
        local_indices = torch.tensor(unique_indices, dtype=torch.long)
        frame_tokens = gallery_frame_tokens[local_indices].to(
            device, non_blocking=True
        )
        frame_embeddings = gallery_frame_embeddings[local_indices].to(
            device, non_blocking=True
        )
        text_tokens = query_token_embeddings[start:end].to(device, non_blocking=True)
        text_embeddings = query_pooled_embeddings[start:end].to(
            device, non_blocking=True
        )
        text_mask = query_attention_masks[start:end].to(device, non_blocking=True)
        with torch.no_grad(), _autocast(device):
            output = model.forward_from_features(
                frame_tokens,
                frame_embeddings,
                text_tokens,
                text_embeddings,
                text_mask,
            )
        local_candidates = torch.tensor(
            [[local_index[int(value)] for value in row] for row in candidate_block.tolist()],
            dtype=torch.long,
        )
        values = output.score_matrix.float().cpu().gather(1, local_candidates)
        reranked = merge_reranked_scores(reranked, candidate_block, values)
        top1_local = local_candidates[:, 0].to(device)
        rows = torch.arange(end - start, device=device)
        top1_maps.append(output.evidence.evidence_map[rows, top1_local].float().cpu())
    return reranked, torch.cat(top1_maps, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--gallery-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--rerank-query-batch-size", type=int, default=4)
    parser.add_argument("--rerank-k", type=int, action="append", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.rerank_k is None:
        args.rerank_k = [20, 50, 100]
    return args


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        worktree = Path(__file__).resolve().parents[1]
        state = git_state(worktree)
        if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
            raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
        release = Path(args.data_release)
        manifest = Path(args.development_manifest)
        checkpoint_path = Path(args.checkpoint)
        for required in (release, manifest, checkpoint_path):
            if not required.exists():
                raise FileNotFoundError(required)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EVALUATION_REQUIRES_CUDA")
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        rows = load_exact_pair_rows(manifest, split="development")
        assert_mask_free(rows)
        pair_rows = canonical_pair_rows(rows)
        positive, ignored = exact_relevance_masks(rows, pair_rows)
        if len(pair_rows) != len(
            {row["canonical_pair_id"] for row in pair_rows}
        ):
            raise RuntimeError("DUPLICATE_GALLERY_IDS")
        processor = AutoProcessor.from_pretrained(
            args.siglip2_model, local_files_only=True
        )
        backbone = Siglip2Backbone(
            args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
        )
        model = Siglip2TemporalRetrievalModel(
            backbone, Siglip2TemporalConfig()
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError("checkpoint lacks model_state")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        backbone.freeze_all()

        started = time.perf_counter()
        gallery_frame_tokens, gallery_frame_embeddings, gallery_embeddings = encode_gallery_features(
            model,
            backbone,
            processor,
            pair_rows,
            device,
            args.gallery_batch_size,
        )
        query_token_embeddings, query_pooled_embeddings, query_attention_masks = encode_queries(
            backbone,
            processor,
            rows,
            device,
            args.query_batch_size,
        )
        query_embeddings = torch.nn.functional.normalize(
            query_pooled_embeddings.float(), dim=-1
        )
        temperature = float(model.retrieval_temperature.detach().cpu())
        raw_global_scores = global_stage_scores(
            query_embeddings, gallery_embeddings, temperature
        )
        # Ignored ambiguity records are excluded from ranking, not treated as
        # hard negatives.  Keep a finite floor so integrity checks remain
        # meaningful and serialized tensors are portable across loaders.
        global_scores = raw_global_scores.masked_fill(ignored, -1.0e4)
        global_metrics = full_gallery_metrics(global_scores, positive)
        rerank_outputs: dict[str, dict[str, Any]] = {}
        max_k = min(max(args.rerank_k), len(pair_rows))
        max_reranked_scores, evidence_maps = rerank_top_k(
            model,
            gallery_frame_tokens,
            gallery_frame_embeddings,
            query_token_embeddings,
            query_pooled_embeddings,
            query_attention_masks,
            global_scores,
            device,
            max_k=max_k,
            query_batch_size=args.rerank_query_batch_size,
        )
        for requested_k in sorted(set(args.rerank_k)):
            k = min(int(requested_k), len(pair_rows))
            candidates = select_topk_candidates(global_scores, k)
            rerank_values = max_reranked_scores.gather(1, candidates)
            scores = merge_reranked_scores(global_scores, candidates, rerank_values)
            rerank_outputs[str(requested_k)] = {
                "k": k,
                "metrics": full_gallery_metrics(scores, positive),
                "scores": scores,
                "candidates": candidates,
            }

        integrity = audit_ranking_integrity(global_scores, rows, pair_rows)
        integrity["development_manifest_sha256"] = _sha256_file(manifest)
        integrity["data_release_sha256"] = sha256_path(release)
        integrity["ordered_query_ids_sha256"] = ordered_id_sha256(rows, "caption_id")
        integrity["ordered_gallery_ids_sha256"] = hashlib.sha256(
            ("\n".join(str(row["canonical_pair_id"]) for row in pair_rows) + "\n").encode()
        ).hexdigest()
        integrity["mask_free"] = True
        full_rankings = {
            "global_scores": global_scores,
            "positive_mask": positive,
            "ignored_mask": ignored,
            "reranked_scores": {
                key: value["scores"] for key, value in rerank_outputs.items()
            },
        }
        torch.save(full_rankings, run / "full_rankings.pt")
        write_jsonl(
            run / "rankings_top100.jsonl",
            ranking_records(max_reranked_scores, rows, pair_rows, top_k=100),
        )
        torch.save(evidence_maps, run / "evidence_maps_top1.pt")
        write_json(
            run / "evaluation_metrics.json",
            {
                "protocol": "QCPR_EXACT_FULL_GALLERY",
                "query_count": len(rows),
                "gallery_count": len(pair_rows),
                "global": global_metrics,
                "reranked": {
                    key: value["metrics"] for key, value in rerank_outputs.items()
                },
                "per_source_global": metrics_by_query_group(
                    global_scores, positive, rows
                ),
                "candidate_recall": {
                    f"candidate_hit_at_{k}": global_metrics[f"candidate_hit_at_{k}"]
                    for k in (10, 50, 100, 500)
                    if f"candidate_hit_at_{k}" in global_metrics
                },
            },
        )
        write_json(run / "ranking_integrity.json", integrity)
        write_json(
            run / "model_contract.json",
            {
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "siglip2_model": str(args.siglip2_model),
                "runtime_class": backbone.runtime_class,
                "native_visual_tokens": list(gallery_frame_tokens.shape),
                "pair_embedding_shape": list(gallery_embeddings.shape),
                "text_embedding_shape": list(query_embeddings.shape),
                "temperature": temperature,
                "rerank_k": sorted(set(args.rerank_k)),
                "mask_access": False,
            },
        )
        write_json(
            run / "runtime_profile.json",
            {
                "total_wall_seconds": time.perf_counter() - started,
                "cpu_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2),
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else None
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / (1024**3)
                    if device.type == "cuda"
                    else None
                ),
                "device": str(device),
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
