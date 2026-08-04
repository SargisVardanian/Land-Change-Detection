#!/usr/bin/env python3
"""Fixed-contract full-data B1 retrieval trainer for QCPR Stage-2.

This entrypoint is deliberately separate from the historical R1 trainer.  It
uses the selected B1 framewise gated-difference head, frozen UniverSat/Jina
backbones, a real 256x128 GradCache logical loss, fixed optimizer steps, and
runtime exposure hashes.  It never opens dense labels or mask fields.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.models.qcpr_single_pass import SinglePassConfig, balanced_siglip_loss
from land_change_detection.training.qcpr_retrieval_repair import (
    LogicalBatchContract,
    exact_grad_cache_backward,
)
from screen_qcpr_stage2_compatible_architectures import (
    FramewiseGatedDifferenceFusion,
    ScreenModel,
    image_batch,
    load_text_batch,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grouped_rows(
    path: Path,
    *,
    allowed_scopes: set[str],
    allowed_sources: set[str] | None,
    allow_generic_no_change: bool,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in read_jsonl(path):
        scope = str(raw.get("query_scope") or "unknown")
        if scope not in allowed_scopes:
            continue
        if scope == "generic_no_change" and not allow_generic_no_change:
            continue
        if raw.get("training_enabled") is False:
            continue
        source = str(raw.get("dataset_name") or raw.get("source_dataset") or "unknown")
        if allowed_sources and source not in allowed_sources:
            continue
        pair_id = str(raw.get("canonical_pair_id") or raw.get("pair_id") or "")
        text = str(raw.get("caption") or raw.get("text") or "").strip()
        if not pair_id or not text:
            continue
        if pair_id not in groups:
            groups[pair_id] = {
                "pair_id": pair_id,
                "dataset_name": source,
                "t1_path": raw.get("t1_path"),
                "t2_path": raw.get("t2_path"),
                "captions": [],
                "caption_ids": [],
                "ignored_pair_ids": [],
                "query_scopes": [],
                "change_status": str(raw.get("change_status") or raw.get("changeflag") or "unknown"),
            }
            order.append(pair_id)
        group = groups[pair_id]
        caption_id = str(raw.get("caption_id") or f"{pair_id}:caption:{len(group['captions'])}")
        if text in group["captions"]:
            continue
        if not group["t1_path"] or not group["t2_path"]:
            raise ValueError(f"pair {pair_id} has incomplete frame paths")
        group["captions"].append(text)
        group["caption_ids"].append(caption_id)
        group["ignored_pair_ids"].append(sorted({str(x) for x in (raw.get("ignored_pair_ids") or [])}))
        group["query_scopes"].append(scope)
    result = [groups[pair_id] for pair_id in order if groups[pair_id]["captions"]]
    if len(result) < 128:
        raise RuntimeError(f"{path} has only {len(result)} usable pairs; need at least 128")
    return result


class PairFeatureStore:
    """Chunked CPU feature store; never materializes the full gallery on GPU."""

    def __init__(self, root: Path, pair_ids: list[str], manifest_sha256: str):
        self.root = root
        meta_path = root / "cache_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing feature cache metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("pair_ids") != pair_ids:
            raise RuntimeError("feature cache pair ordering does not match manifest")
        if self.meta.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("feature cache manifest SHA256 mismatch")
        self.entries = self.meta["entries"]
        self.loaded: collections.OrderedDict[str, Tensor] = collections.OrderedDict()

    def _load_chunk(self, name: str) -> Tensor:
        if name not in self.loaded:
            payload = torch.load(self.root / name, map_location="cpu", weights_only=False)
            features = payload["features"] if isinstance(payload, dict) else payload
            self.loaded[name] = features
            while len(self.loaded) > 4:
                self.loaded.popitem(last=False)
        else:
            self.loaded.move_to_end(name)
        return self.loaded[name]

    def get(self, indices: list[int]) -> Tensor:
        values: list[Tensor] = []
        for index in indices:
            entry = self.entries[str(index)]
            values.append(self._load_chunk(entry["chunk"])[entry["offset"]])
        return torch.stack(values, dim=0)


@torch.inference_mode()
def extract_framewise_cache(
    rows: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    cache_root: Path,
    universat_source: str,
    universat_checkpoint: str,
    image_size: int,
    extract_batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in rows]
    cache_root.mkdir(parents=True, exist_ok=True)
    meta_path = cache_root / "cache_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pair_ids") != pair_ids or meta.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"existing cache is incompatible: {cache_root}")
        return meta

    backend = UniverSatJointBackend(
        UniverSatBackendConfig(
            source_dir=universat_source,
            checkpoint_dir=universat_checkpoint,
            output_grid=32,
            freeze=True,
        )
    ).to(device).eval()
    frame_backend = UniverSatFrameBackend(backend.model, output_grid=32, visual_dim=768).to(device).eval()
    encoder = SequenceUniverSatEncoder(
        frame_backend,
        output_grid=32,
        visual_dim=768,
        freeze=True,
    ).to(device).eval()

    entries: dict[str, dict[str, Any]] = {}
    chunk_hashes: dict[str, str] = {}
    started = time.monotonic()
    peak_allocated = 0
    peak_reserved = 0
    chunk_index = 0
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        parts: list[Tensor] = []
        for batch_start in range(start, stop, extract_batch_size):
            batch_stop = min(batch_start + extract_batch_size, stop)
            batch_indices = list(range(batch_start, batch_stop))
            batch = image_batch(rows, batch_indices, image_size).to(device)
            output = encoder(batch)
            features = output.features if hasattr(output, "features") else output
            if tuple(features.shape[1:]) != (2, 1024, 768):
                raise RuntimeError(f"B1 native framewise feature shape mismatch: {tuple(features.shape)}")
            parts.append(features.detach().to(device="cpu", dtype=torch.float16))
        chunk_features = torch.cat(parts, dim=0).contiguous()
        chunk_name = f"features_{chunk_index:05d}.pt"
        chunk_path = cache_root / chunk_name
        torch.save({"features": chunk_features}, chunk_path)
        chunk_hashes[chunk_name] = sha256_file(chunk_path)
        for offset, global_index in enumerate(range(start, stop)):
            entries[str(global_index)] = {"chunk": chunk_name, "offset": offset}
        chunk_index += 1
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        del chunk_features, parts
        torch.cuda.empty_cache()

    meta = {
        "schema_version": "qcpr-stage2-b1-framewise-feature-cache-v1",
        "pair_ids": pair_ids,
        "manifest_sha256": manifest_sha256,
        "feature_shape": [2, 1024, 768],
        "image_size": image_size,
        "chunk_size": chunk_size,
        "entries": entries,
        "chunk_hashes": chunk_hashes,
        "pair_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": peak_allocated / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_initial_b1(path: Path, device: torch.device) -> tuple[ScreenModel, SinglePassConfig, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "B1":
        raise RuntimeError(f"expected a B1 screen checkpoint, got {payload.get('kind')}")
    config_keys = set(SinglePassConfig.__dataclass_fields__)
    config_data = {key: value for key, value in (payload.get("config") or {}).items() if key in config_keys}
    cfg = SinglePassConfig(**config_data)
    model = ScreenModel("B1", cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, cfg, sha256_file(path)


def make_schedule(pair_count: int, logical_batch: int, steps: int, seed: int) -> list[list[int]]:
    if pair_count < logical_batch:
        raise ValueError("pair count is smaller than the logical batch")
    schedule: list[list[int]] = []
    for step in range(steps):
        order = list(range(pair_count))
        random.Random(seed + step * 1009).shuffle(order)
        start = (step * logical_batch) % pair_count
        schedule.append([order[(start + offset) % pair_count] for offset in range(logical_batch)])
    return schedule


def choose_caption(row: dict[str, Any], step: int, slot: int) -> tuple[str, str, set[str], str]:
    count = len(row["captions"])
    caption_index = (step + slot) % count
    text = str(row["captions"][caption_index])
    caption_id = str(row["caption_ids"][caption_index])
    ignored = {str(value) for value in row["ignored_pair_ids"][caption_index]}
    scope = str(row["query_scopes"][caption_index])
    return text, caption_id, ignored, scope


def make_logical_batches(
    rows: list[dict[str, Any]],
    store: PairFeatureStore,
    pair_indices: list[int],
    step: int,
    physical_microbatch: int,
    captions_per_pair: int,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[set[str]], list[str]]:
    if len(pair_indices) % physical_microbatch:
        raise ValueError("logical pair batch must be divisible by physical microbatch")
    logical_captions: list[str] = []
    query_ids: list[str] = []
    ignored_sets: list[set[str]] = []
    query_scopes: list[str] = []
    for pair_index in pair_indices:
        row = rows[pair_index]
        for slot in range(captions_per_pair):
            text, caption_id, ignored, scope = choose_caption(row, step, slot)
            logical_captions.append(text)
            query_ids.append(caption_id)
            ignored_sets.append(ignored)
            query_scopes.append(scope)
    micro_batches: list[dict[str, Any]] = []
    for start in range(0, len(pair_indices), physical_microbatch):
        indices = pair_indices[start : start + physical_microbatch]
        text_start = start * captions_per_pair
        text_stop = (start + len(indices)) * captions_per_pair
        micro_batches.append({
            "features": store.get(indices),
            "captions": logical_captions[text_start:text_stop],
        })
    return micro_batches, logical_captions, query_ids, ignored_sets, query_scopes


def make_supervision(
    pair_ids: list[str],
    query_pair_ids: list[str],
    ignored_sets: list[set[str]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    positive = torch.zeros((len(query_pair_ids), len(pair_ids)), dtype=torch.bool, device=device)
    excluded = torch.zeros_like(positive)
    columns = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    for query_index, pair_id in enumerate(query_pair_ids):
        positive[query_index, columns[pair_id]] = True
        for ignored_pair in ignored_sets[query_index]:
            column = columns.get(ignored_pair)
            if column is not None and ignored_pair != pair_id:
                excluded[query_index, column] = True
    return positive, excluded


def encode_microbatch(
    model: ScreenModel,
    text_encoder: JinaV5TextEncoder,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    base, tokens, attention, content = load_text_batch(text_encoder, batch["captions"], device)
    text = model._text_embedding(base, tokens, attention, content)
    visual = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
    pair = model.pair_embeddings(visual)
    return text, pair


@torch.no_grad()
def evaluate(
    model: ScreenModel,
    text_encoder: JinaV5TextEncoder,
    rows: list[dict[str, Any]],
    store: PairFeatureStore,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    model.eval()
    pair_parts: list[Tensor] = []
    for start in range(0, len(rows), 32):
        pair_parts.append(model.pair_embeddings(store.get(list(range(start, min(start + 32, len(rows))))).to(device, dtype=torch.float32)).cpu())
    pair_vectors = torch.cat(pair_parts, dim=0).to(device)
    queries: list[tuple[int, str, str]] = []
    for pair_index, row in enumerate(rows):
        for caption_index, text in enumerate(row["captions"]):
            queries.append((pair_index, str(text), str(row["caption_ids"][caption_index])))
    ranks: list[int] = []
    ranking_rows: list[dict[str, Any]] = []
    for start in range(0, len(queries), 64):
        chunk = queries[start : start + 64]
        base, tokens, attention, content = load_text_batch(text_encoder, [item[1] for item in chunk], device)
        text_vectors = model._text_embedding(base, tokens, attention, content)
        scores = text_vectors @ pair_vectors.T
        order = scores.argsort(dim=1, descending=True)
        for local, (true_index, text, query_id) in enumerate(chunk):
            rank = int((order[local] == true_index).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            top = order[local, :10].tolist()
            ranking_rows.append({
                "query_id": query_id,
                "text": text,
                "true_pair_id": rows[true_index]["pair_id"],
                "true_pair_rank": rank,
                "top10_pair_ids": [rows[index]["pair_id"] for index in top],
                "top10_scores": [float(scores[local, index]) for index in top],
            })
    ranks_tensor = torch.tensor(ranks, dtype=torch.float64)
    metrics = {
        "query_count": len(ranks),
        "gallery_pair_count": len(rows),
        "recall_at_1": float((ranks_tensor <= 1).double().mean()),
        "recall_at_5": float((ranks_tensor <= 5).double().mean()),
        "recall_at_10": float((ranks_tensor <= 10).double().mean()),
        "recall_at_50": float((ranks_tensor <= 50).double().mean()),
        "recall_at_100": float((ranks_tensor <= 100).double().mean()),
        "mrr": float((1.0 / ranks_tensor).mean()),
        "mean_rank": float(ranks_tensor.mean()),
        "median_rank": float(ranks_tensor.median()),
        "metric_contract": "exact_single_positive_diagnostic; generic_no_change excluded upstream",
    }
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in ranking_rows) + "\n", encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("P1", "P2-real", "P2-semantic"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--universat-source", required=True)
    parser.add_argument("--universat-checkpoint", required=True)
    parser.add_argument("--jina-model", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--steps", type=int, default=348)
    parser.add_argument("--physical-microbatch", type=int, default=16)
    parser.add_argument("--logical-physical-batch", type=int, default=128)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--feature-chunk-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--allow-generic-no-change", action="store_true")
    parser.add_argument("--source-allowlist", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stage == "P2-semantic":
        raise SystemExit("P2-semantic is disabled until reviewed semantic gold rows are non-empty and a graded semantic objective is implemented")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 B1 trainer requires CUDA")
    if args.steps <= 0 or args.logical_physical_batch % args.physical_microbatch:
        raise ValueError("invalid fixed-step or logical/microbatch contract")
    if args.captions_per_pair != 2 or args.physical_microbatch != 16 or args.logical_physical_batch != 128:
        raise ValueError("Stage-2 initial contract is fixed at captions=2, microbatch=16, logical=128")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    allowed_sources = {value.strip() for value in args.source_allowlist.split(",") if value.strip()} or None
    scopes = {"exact_pair"} | ({"generic_no_change"} if args.allow_generic_no_change else set())
    train_rows = grouped_rows(
        args.train_manifest,
        allowed_scopes=scopes,
        allowed_sources=allowed_sources,
        allow_generic_no_change=args.allow_generic_no_change,
    )
    dev_rows = grouped_rows(
        args.development_manifest,
        allowed_scopes=scopes,
        allowed_sources=allowed_sources,
        allow_generic_no_change=args.allow_generic_no_change,
    )
    train_manifest_sha = sha256_file(args.train_manifest)
    dev_manifest_sha = sha256_file(args.development_manifest)
    train_cache = extract_framewise_cache(
        train_rows,
        manifest_sha256=train_manifest_sha,
        cache_root=args.cache_root / "train",
        universat_source=args.universat_source,
        universat_checkpoint=args.universat_checkpoint,
        image_size=args.image_size,
        extract_batch_size=args.extract_batch_size,
        chunk_size=args.feature_chunk_size,
        device=device,
    )
    dev_cache = extract_framewise_cache(
        dev_rows,
        manifest_sha256=dev_manifest_sha,
        cache_root=args.cache_root / "development",
        universat_source=args.universat_source,
        universat_checkpoint=args.universat_checkpoint,
        image_size=args.image_size,
        extract_batch_size=args.extract_batch_size,
        chunk_size=args.feature_chunk_size,
        device=device,
    )
    train_store = PairFeatureStore(args.cache_root / "train", [row["pair_id"] for row in train_rows], train_manifest_sha)
    dev_store = PairFeatureStore(args.cache_root / "development", [row["pair_id"] for row in dev_rows], dev_manifest_sha)
    model, cfg, initial_sha = load_initial_b1(args.initial_checkpoint, device)
    text_encoder = JinaV5TextEncoder(
        JinaV5TextConfig(
            model_path=args.jina_model,
            max_length=256,
            global_projection_mode="matryoshka_truncate",
            freeze=True,
        )
    ).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    total_steps = args.steps
    warmup = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (max(step, 1) / warmup) if step < warmup else 0.5 * (1.0 + math.cos(math.pi * min(step - warmup, total_steps - warmup) / max(total_steps - warmup, 1))),
    )
    contract = LogicalBatchContract(args.physical_microbatch, args.logical_physical_batch, args.captions_per_pair)
    schedule = make_schedule(len(train_rows), args.logical_physical_batch, args.steps, args.seed)
    pair_presentations: collections.Counter[str] = collections.Counter()
    query_presentations: collections.Counter[str] = collections.Counter()
    source_presentations: collections.Counter[str] = collections.Counter()
    scope_presentations: collections.Counter[str] = collections.Counter()
    per_step_hashes: list[str] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    batch_contract_path = args.output_dir / "batch_contract.json"
    first_batch_checked = False
    first_batch_contract: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    for step, pair_indices in enumerate(schedule):
        micro_batches, captions, query_ids, ignored_sets, scopes_for_queries = make_logical_batches(
            train_rows,
            train_store,
            pair_indices,
            step,
            args.physical_microbatch,
            args.captions_per_pair,
        )
        pair_ids = [train_rows[index]["pair_id"] for index in pair_indices]
        query_pair_ids = [pair_ids[index // args.captions_per_pair] for index in range(len(query_ids))]
        positive, excluded = make_supervision(pair_ids, query_pair_ids, ignored_sets, device)
        if positive.shape != contract.score_matrix_shape:
            raise RuntimeError(f"supervision shape mismatch: {tuple(positive.shape)}")
        optimizer.zero_grad(set_to_none=True)

        def encode(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
            return encode_microbatch(model, text_encoder, batch, device)

        def loss_function(text_vectors: Tensor, pair_vectors: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
            logits = model.logit_scale.exp().clamp(1e-3, 100.0) * (text_vectors @ pair_vectors.T) + model.logit_bias
            return balanced_siglip_loss(logits, positive, ~positive & ~excluded)

        loss, stats, shape = exact_grad_cache_backward(
            micro_batches=micro_batches,
            encode=encode,
            loss_function=loss_function,
            device=device,
            autocast_factory=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
        )
        if shape != contract.score_matrix_shape:
            raise RuntimeError(f"logical score matrix mismatch: {shape}")
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"non-finite loss or gradients at step {step + 1}")
        if not first_batch_checked:
            first_batch_contract = {
                "score_matrix_shape": list(shape),
                "logical_physical_batch": args.logical_physical_batch,
                "logical_text_queries": contract.logical_query_count,
                "physical_microbatch": args.physical_microbatch,
                "captions_per_pair": args.captions_per_pair,
                "gradcache_recomputation": True,
                "finite_loss": bool(torch.isfinite(loss)),
                "finite_gradients": True,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            batch_contract_path.write_text(json.dumps(first_batch_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            first_batch_checked = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        for pair_index in pair_indices:
            row = train_rows[pair_index]
            pair_presentations[row["pair_id"]] += 1
            source_presentations[row["dataset_name"]] += 1
        for query_id, scope in zip(query_ids, scopes_for_queries, strict=True):
            query_presentations[query_id] += 1
            scope_presentations[scope] += 1
        step_schedule = {
            "step": step + 1,
            "pair_ids": pair_ids,
            "query_ids": query_ids,
        }
        per_step_hashes.append(stable_sha(step_schedule))
        record = {
            "step": step + 1,
            "loss": float(loss),
            "positive_loss": float(stats["positive_loss"]),
            "negative_loss": float(stats["negative_loss"]),
            "positive_similarity": float(stats["positive_similarity"]),
            "negative_similarity": float(stats["negative_similarity"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if args.eval_every and (step + 1) % args.eval_every == 0:
            evaluation = evaluate(model, text_encoder, dev_rows, dev_store, device, args.output_dir / f"rankings_step_{step + 1:04d}.jsonl")
            (args.output_dir / f"evaluation_step_{step + 1:04d}.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not first_batch_checked or len(schedule) != args.steps:
        raise RuntimeError("first-batch or fixed-step contract did not complete")
    final_evaluation = evaluate(model, text_encoder, dev_rows, dev_store, device, args.output_dir / "rankings_final.jsonl")
    (args.output_dir / "evaluation_final.json").write_text(json.dumps(final_evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pair_sequence = [pair_id for step in schedule for pair_id in [train_rows[index]["pair_id"] for index in step]]
    query_sequence = []
    for step, pair_indices in enumerate(schedule):
        for pair_index in pair_indices:
            row = train_rows[pair_index]
            for slot in range(args.captions_per_pair):
                query_sequence.append(choose_caption(row, step, slot)[1])
    exposure = {
        "schema_version": "qcpr-stage2-b1-exposure-v1",
        "total_pair_presentations": len(pair_sequence),
        "total_query_presentations": len(query_sequence),
        "unique_pairs": len(set(pair_sequence)),
        "unique_queries": len(set(query_sequence)),
        "mean_presentations_per_pair": len(pair_sequence) / max(len(set(pair_sequence)), 1),
        "mean_presentations_per_query": len(query_sequence) / max(len(set(query_sequence)), 1),
        "presentations_per_pair": dict(sorted(pair_presentations.items())),
        "presentations_per_query": dict(sorted(query_presentations.items())),
        "per_source_presentations": dict(sorted(source_presentations.items())),
        "query_scope_presentations": dict(sorted(scope_presentations.items())),
        "pair_sequence_sha256": stable_sha(pair_sequence),
        "query_sequence_sha256": stable_sha(query_sequence),
        "per_step_schedule_sha256": per_step_hashes,
        "steps": args.steps,
        "logical_batch_contract": {
            "physical_microbatch": args.physical_microbatch,
            "logical_physical_batch": args.logical_physical_batch,
            "captions_per_pair": args.captions_per_pair,
            "logical_text_queries": contract.logical_query_count,
            "score_matrix": list(contract.score_matrix_shape),
        },
    }
    (args.output_dir / "exposure_accounting.json").write_text(json.dumps(exposure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    batch_contract = {
        **first_batch_contract,
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "feature_cache_train": train_cache,
        "feature_cache_development": dev_cache,
    }
    batch_contract_path.write_text(json.dumps(batch_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = {
        "schema_version": "qcpr-stage2-b1-checkpoint-v1",
        "stage": args.stage,
        "model_kind": "B1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": args.steps,
        "seed": args.seed,
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_sha256": initial_sha,
        "train_manifest_sha256": train_manifest_sha,
        "development_manifest_sha256": dev_manifest_sha,
        "config": cfg.__dict__,
        "fixed_contract": first_batch_contract,
        "exposure_schedule_sha256": exposure["pair_sequence_sha256"],
    }
    torch.save(checkpoint, args.output_dir / "checkpoint_final.pt")
    (args.output_dir / "training_complete.json").write_text(json.dumps({
        "status": "COMPLETED_FIXED_STEPS",
        "stage": args.stage,
        "global_step": args.steps,
        "requested_steps": args.steps,
        "elapsed_seconds": time.monotonic() - started,
        "evaluation": final_evaluation,
        "checkpoint": str(args.output_dir / "checkpoint_final.pt"),
        "checkpoint_sha256": sha256_file(args.output_dir / "checkpoint_final.pt"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
