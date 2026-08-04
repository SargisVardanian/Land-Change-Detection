#!/usr/bin/env python3
"""Frozen B1 evaluation on the immutable common exact development gallery."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from screen_qcpr_stage2_compatible_architectures import ScreenModel, group_rows, image_batch, load_text_batch
from land_change_detection.models.qcpr_single_pass import SinglePassConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_rows_preserve_queries(rows: list[dict]) -> list[dict]:
    """Group physical pairs without deduplicating query records.

    The common frozen gallery is defined over the immutable 9,640 query rows.
    Several pairs contain repeated caption text, and those rows remain valid
    query exposures for this audit even though training samplers may dedupe
    text within a physical pair.
    """
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for index, raw in enumerate(rows):
        pair_id = str(raw.get("canonical_pair_id") or raw.get("pair_id") or "")
        text = str(raw.get("caption") or raw.get("text") or "").strip()
        if not pair_id or not text:
            continue
        if pair_id not in grouped:
            grouped[pair_id] = {
                "pair_id": pair_id,
                "dataset_name": raw.get("dataset_name", "unknown"),
                "t1_path": raw["t1_path"],
                "t2_path": raw["t2_path"],
                "captions": [],
                "caption_ids": [],
                "query_scopes": [],
            }
            order.append(pair_id)
        row = grouped[pair_id]
        row["captions"].append(text)
        row["caption_ids"].append(str(raw.get("caption_id") or f"{pair_id}:query:{index}"))
        row["query_scopes"].append(str(raw.get("query_scope") or "unknown"))
    return [grouped[pair_id] for pair_id in order]


@torch.inference_mode()
def extract_features(rows: list[dict], args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    cache_path = args.feature_cache
    if cache_path and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("pair_ids") != [row["pair_id"] for row in rows]:
            raise RuntimeError("common evaluation feature cache pair order mismatch")
        return payload["features"]
    backend = UniverSatJointBackend(
        UniverSatBackendConfig(
            source_dir=args.universat_source,
            checkpoint_dir=args.universat_checkpoint,
            output_grid=32,
            freeze=True,
        )
    ).to(device).eval()
    frame_backend = UniverSatFrameBackend(backend.model, output_grid=32, visual_dim=768).to(device).eval()
    encoder = SequenceUniverSatEncoder(frame_backend, output_grid=32, visual_dim=768, freeze=True).to(device).eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(rows), args.extract_batch_size):
        batch = image_batch(rows, list(range(start, min(start + args.extract_batch_size, len(rows)))), args.image_size).to(device)
        output = encoder(batch)
        features = output.features if hasattr(output, "features") else output
        if tuple(features.shape[1:]) != (2, 1024, 768):
            raise RuntimeError(f"B1 common feature shape mismatch: {tuple(features.shape)}")
        chunks.append(features.detach().to(device="cpu", dtype=torch.float16))
    features = torch.cat(chunks, dim=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": "qcpr-stage2-common-b1-features-v1", "pair_ids": [row["pair_id"] for row in rows], "features": features}, cache_path)
    return features


@torch.inference_mode()
def evaluate(model: ScreenModel, text_encoder: JinaV5TextEncoder, rows: list[dict], features: torch.Tensor, device: torch.device, text_batch: int, output_dir: Path) -> dict:
    model.eval()
    pair_parts = []
    for start in range(0, len(rows), 64):
        pair_parts.append(model.pair_embeddings(features[start:start + 64].to(device, dtype=torch.float32)).cpu())
    pair_embeddings = torch.cat(pair_parts, dim=0).to(device)
    captions: list[str] = []
    mapping: list[int] = []
    query_ids: list[str] = []
    query_scopes: list[str] = []
    for pair_index, row in enumerate(rows):
        for caption_index, caption in enumerate(row["captions"]):
            captions.append(str(caption))
            mapping.append(pair_index)
            query_ids.append(str(row["caption_ids"][caption_index]))
            query_scopes.append(str(row["query_scopes"][caption_index]))
    rank_indices: list[torch.Tensor] = []
    rank_values: list[int] = []
    top100: list[dict] = []
    for start in range(0, len(captions), text_batch):
        chunk = captions[start:start + text_batch]
        base, tokens, attention, content = load_text_batch(text_encoder, chunk, device)
        text_embeddings = model._text_embedding(base, tokens, attention, content)
        scores = model.logit_scale.exp().clamp(1e-3, 100.0) * (text_embeddings @ pair_embeddings.T) + model.logit_bias
        order = torch.argsort(scores, dim=1, descending=True)
        rank_indices.append(order.cpu().to(torch.int32))
        for local, pair_index in enumerate(mapping[start:start + len(chunk)]):
            rank = int((order[local] == pair_index).nonzero(as_tuple=False)[0].item()) + 1
            rank_values.append(rank)
            top = order[local, :100].tolist()
            top100.append({
                "query_id": query_ids[start + local],
                "query_scope": query_scopes[start + local],
                "text": captions[start + local],
                "true_pair_id": rows[pair_index]["pair_id"],
                "true_pair_rank": rank,
                "top100_pair_ids": [rows[index]["pair_id"] for index in top],
                "top100_scores": [float(scores[local, index]) for index in top],
            })
    ranks = torch.tensor(rank_values, dtype=torch.float64)
    def summarize(values: list[int], scope: str) -> dict:
        subset = torch.tensor(values, dtype=torch.float64)
        if not len(values):
            return {"query_count": 0, "scope": scope}
        return {
            "query_count": len(values),
            "scope": scope,
            "gallery_pair_count": len(rows),
            "recall_at_1": float((subset <= 1).double().mean()),
            "recall_at_5": float((subset <= 5).double().mean()),
            "recall_at_10": float((subset <= 10).double().mean()),
            "recall_at_50": float((subset <= 50).double().mean()),
            "recall_at_100": float((subset <= 100).double().mean()),
            "mrr": float((1.0 / subset).mean()),
            "mean_rank": float(subset.mean()),
            "median_rank": float(subset.median()),
        }
    exact_values = [rank for rank, scope in zip(rank_values, query_scopes, strict=True) if scope == "exact_pair"]
    generic_values = [rank for rank, scope in zip(rank_values, query_scopes, strict=True) if scope == "generic_no_change"]
    metrics = {
        "exact_primary": summarize(exact_values, "exact_pair"),
        "generic_no_change_diagnostic": summarize(generic_values, "generic_no_change"),
        "all_diagnostic": summarize(rank_values, "all_queries"),
        "metric_contract": "QCPR_EXACT_FULL_GALLERY; exact_pair is primary; generic_no_change is diagnostic only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"query_ids": query_ids, "query_scopes": query_scopes, "pair_ids": [row["pair_id"] for row in rows], "rank_indices": torch.cat(rank_indices), "true_pair_indices": torch.tensor(mapping, dtype=torch.int32), "true_pair_ranks": torch.tensor(rank_values, dtype=torch.int32)}, output_dir / "full_rankings.pt")
    (output_dir / "rankings_top100.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in top100) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--universat-source", required=True)
    parser.add_argument("--universat-checkpoint", required=True)
    parser.add_argument("--jina-model", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--text-batch", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("common B1 evaluation requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_float32_matmul_precision("high")
    raw = read_rows(args.development_manifest)
    rows = group_rows_preserve_queries(raw)
    if len(rows) != 1928:
        raise RuntimeError(f"common development gallery must contain 1928 pairs, got {len(rows)}")
    if sum(len(row["captions"]) for row in rows) != 9640:
        raise RuntimeError("common development query count must be 9640")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_kind = payload.get("kind") or payload.get("model_kind")
    if checkpoint_kind != "B1":
        raise RuntimeError("common evaluator requires a B1 checkpoint")
    config_keys = set(SinglePassConfig.__dataclass_fields__)
    cfg = SinglePassConfig(**{key: value for key, value in (payload.get("config") or {}).items() if key in config_keys})
    model = ScreenModel("B1", cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    text_encoder = JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=256, global_projection_mode="matryoshka_truncate", freeze=True)).to(device).eval()
    started = time.monotonic()
    features = extract_features(rows, args, device)
    metrics = evaluate(model, text_encoder, rows, features, device, args.text_batch, args.output_dir)
    state = {
        "status": "PASS_FULL_RANKINGS",
        "metric_protocol": "QCPR_EXACT_FULL_GALLERY",
        "development_manifest": str(args.development_manifest),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "query_count": 9640,
        "physical_pair_count": 1928,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "feature_cache": str(args.feature_cache),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "full_rankings": str(args.output_dir / "full_rankings.pt"),
        "full_rankings_sha256": sha256_file(args.output_dir / "full_rankings.pt"),
        "top100_rankings": str(args.output_dir / "rankings_top100.jsonl"),
        "top100_rankings_sha256": sha256_file(args.output_dir / "rankings_top100.jsonl"),
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "backbones_frozen": True,
        "mask_access": False,
    }
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps({"metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "common_frozen_evaluation_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
