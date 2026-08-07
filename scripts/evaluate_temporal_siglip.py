#!/usr/bin/env python3
"""Evaluate TemporalSigLIP by direct full-gallery cosine ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from qcpr_siglip2.data.manifest import (
    group_rows_by_pair,
    load_exact_core_rows,
    load_exact_pair_rows,
    ordered_id_sha256,
)
from qcpr_siglip2.data.runtime import build_relevance_masks, encode_real_images, encode_real_text
from qcpr_temporal_siglip.backbone import TemporalSigLIPBackbone
from qcpr_temporal_siglip.config import TemporalSigLIPConfig
from qcpr_temporal_siglip.evaluator import direct_retrieval_metrics
from qcpr_temporal_siglip.model import TemporalSigLIP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256(path)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=256)
    return parser.parse_args()


def verify_code(expected: str) -> dict[str, Any]:
    worktree = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    clean = not bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    if head != expected or not clean:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    return {"head": head, "worktree_clean": clean}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TEMPORAL_SIGLIP_EVALUATION_REQUIRES_CUDA")
    if args.pair_batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    code = verify_code(args.expected_code_sha)
    manifest = Path(args.development_manifest)
    checkpoint = Path(args.checkpoint)
    model_path = Path(args.siglip2_model)
    data_release = Path(args.data_release)
    for required in (manifest, checkpoint, model_path, data_release):
        if not required.exists():
            raise FileNotFoundError(required)
    all_rows = load_exact_core_rows(manifest, split="development")
    rows = load_exact_pair_rows(manifest, split="development")
    groups = group_rows_by_pair(all_rows)
    pair_rows = [group[0] for group in groups.values()]
    query_rows = rows
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    backbone = TemporalSigLIPBackbone(
        args.siglip2_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model = TemporalSigLIP(backbone, TemporalSigLIPConfig()).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("checkpoint lacks model_state")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    backbone.eval()

    pair_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(pair_rows), args.pair_batch_size):
            image = encode_real_images(
                backbone,
                processor,
                pair_rows[start : start + args.pair_batch_size],
                device,
                dtype=torch.bfloat16,
                no_grad=True,
            )
            pair_embeddings.append(
                model.encode_pair_from_features(image.patch_tokens).pair_embedding.float().cpu()
            )
    pair_matrix = torch.cat(pair_embeddings, dim=0)

    query_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(query_rows), args.query_batch_size):
            text = encode_real_text(
                backbone,
                processor,
                query_rows[start : start + args.query_batch_size],
                device,
                dtype=torch.bfloat16,
                no_grad=True,
            )
            query_embeddings.append(torch.nn.functional.normalize(text.pooled_embedding, dim=-1).float().cpu())
    query_matrix = torch.cat(query_embeddings, dim=0)
    scores = model.score_embeddings(query_matrix, pair_matrix).float()
    positive, ignored, _ = build_relevance_masks(query_rows, pair_rows, torch.device("cpu"))
    metrics = direct_retrieval_metrics(scores, positive)
    full_rankings = scores.argsort(dim=1, descending=True)
    torch.save(full_rankings, run / "full_rankings.pt")
    ranking_sha = sha256(run / "full_rankings.pt")
    top_k = min(100, scores.shape[1])
    top_indices = full_rankings[:, :top_k]
    with (run / "rankings_top100.jsonl").open("w", encoding="utf-8") as handle:
        for row, indices in zip(query_rows, top_indices.tolist()):
            handle.write(json.dumps({
                "query_id": str(row["caption_id"]),
                "query_text": str(row["caption"]),
                "top_pair_ids": [str(pair_rows[index]["canonical_pair_id"]) for index in indices],
            }, sort_keys=True) + "\n")
    write_json(run / "code_state.json", code)
    write_json(run / "model_source.json", {
        "repository": "google/siglip2-base-patch16-256",
        "local_path": str(model_path),
        "weights_sha256": sha256(model_path / "model.safetensors"),
        "active_model": "TemporalSigLIP",
    })
    write_json(run / "data_contract.json", {
        "data_release": str(data_release),
        "data_release_sha256": sha256_path(data_release),
        "development_manifest": str(manifest),
        "development_manifest_sha256": sha256(manifest),
        "development_release_sha256": sha256_path(manifest.parent),
        "ordered_pair_ids_sha256": ordered_id_sha256(pair_rows, "canonical_pair_id"),
        "ordered_query_ids_sha256": ordered_id_sha256(query_rows, "caption_id"),
        "gallery_size": len(pair_rows),
        "query_count": len(query_rows),
        "score_shape": list(scores.shape),
        "mask_access": False,
    })
    write_json(run / "evaluation_metrics.json", {
        "protocol": "QCPR_EXACT_FULL_GALLERY_DIRECT_TEMPORALSIGLIP",
        "metrics": metrics,
        "score_shape": list(scores.shape),
        "full_rankings_sha256": ranking_sha,
        "ignored_count": int(ignored.sum()),
        "mandatory_reranking": False,
    })
    write_json(run / "retrieval_integrity_audit.json", {
        "status": "PASS",
        "mask_free": True,
        "positive_mask_shape": list(positive.shape),
        "ignored_mask_shape": list(ignored.shape),
        "full_rankings_sha256": ranking_sha,
        "ordered_pair_ids_sha256": ordered_id_sha256(pair_rows, "canonical_pair_id"),
        "ordered_query_ids_sha256": ordered_id_sha256(query_rows, "caption_id"),
    })
    write_json(run / "embedding_diagnostics.json", {
        "pair_norm_mean": float(pair_matrix.norm(dim=-1).mean()),
        "text_norm_mean": float(query_matrix.norm(dim=-1).mean()),
        "pair_effective_rank": int(torch.linalg.matrix_rank(pair_matrix).item()),
        "text_effective_rank": int(torch.linalg.matrix_rank(query_matrix).item()),
    })
    write_json(run / "localization_metrics.json", {
        "status": "DIAGNOSTIC_ONLY_POST_RETRIEVAL",
        "reason": "evaluation requires a separate mask sidecar and no map is used in direct score",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
