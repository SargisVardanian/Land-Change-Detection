#!/usr/bin/env python3
"""Bounded Stage-2 compatible temporal architecture screen.

This is not P2 training.  UniverSat and Jina are frozen.  Only the temporal
fusion, PAIR adapter, text adapter and retrieval projections are optimized for
a fixed, small screen budget.  B0/B1/B2 use one common pair/query schedule
and one common development gallery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
from land_change_detection.models.qcpr_single_pass import (
    DeepResidualPairAdapter,
    SinglePassConfig,
    TextSearchProjection,
    balanced_siglip_loss,
)
from land_change_detection.models.qcpr_single_pass_factory import JointUniverSatSeriesEncoder
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.data.unichange_mci import _load_rgb


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_rows(rows: list[dict[str, Any]], max_pairs: int | None = None) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        pair_id = str(row.get("canonical_pair_id") or row.get("pair_id"))
        if pair_id not in groups:
            groups[pair_id] = {
                "pair_id": pair_id,
                "dataset_name": row.get("dataset_name", "unknown"),
                "t1_path": row["t1_path"],
                "t2_path": row["t2_path"],
                "captions": [],
                "query_scopes": [],
            }
            order.append(pair_id)
        text = str(row.get("caption") or row.get("text") or "").strip()
        if text and text not in groups[pair_id]["captions"]:
            groups[pair_id]["captions"].append(text)
            groups[pair_id]["query_scopes"].append(str(row.get("query_scope", "unknown")))
    result = [groups[key] for key in order if groups[key]["captions"]]
    if max_pairs is not None:
        result = result[: max(0, int(max_pairs))]
    if not result:
        raise RuntimeError(f"no grouped pairs in {len(rows)} rows")
    return result


def image_batch(rows: list[dict[str, Any]], indices: list[int], size: int) -> Tensor:
    tensors = []
    for index in indices:
        row = rows[index]
        tensors.append(torch.stack([_load_rgb(row["t1_path"], size), _load_rgb(row["t2_path"], size)], dim=0))
    return torch.stack(tensors, dim=0)


class FramewiseGatedDifferenceFusion(nn.Module):
    """B1: frozen framewise tokens plus learned gated temporal difference."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.context_norm = nn.LayerNorm(dim)
        self.delta_norm = nn.LayerNorm(dim)
        self.magnitude_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, frames: Tensor, text: Tensor | None = None) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 2:
            raise ValueError("framewise features must be [B,2,N,D]")
        before, after = frames[:, 0], frames[:, 1]
        context = 0.5 * (before + after)
        delta = after - before
        magnitude = delta.abs()
        interaction = before * after
        gate = self.gate(torch.cat((self.context_norm(context), self.delta_norm(delta)), dim=-1))
        mixed = self.mix(torch.cat((context, delta, magnitude, interaction), dim=-1))
        return context + gate * delta + mixed


class TextConditionedTemporalFusion(nn.Module):
    """B2: frozen framewise tokens with a query-conditioned temporal gate."""

    def __init__(self, dim: int = 768, text_dim: int = 512):
        super().__init__()
        self.context_norm = nn.LayerNorm(dim)
        self.delta_norm = nn.LayerNorm(dim)
        self.text_projection = nn.Linear(text_dim, dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, frames: Tensor, text: Tensor) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 2:
            raise ValueError("framewise features must be [B,2,N,D]")
        if text.ndim != 2 or text.shape[0] != frames.shape[0]:
            raise ValueError("B2 text must be [B,512]")
        before, after = frames[:, 0], frames[:, 1]
        context = 0.5 * (before + after)
        delta = after - before
        magnitude = delta.abs()
        interaction = before * after
        condition = self.text_projection(text).unsqueeze(1).expand(-1, frames.shape[2], -1)
        gate = self.gate(torch.cat((self.context_norm(context), self.delta_norm(delta), condition), dim=-1))
        mixed = self.mix(torch.cat((context, delta, magnitude, interaction), dim=-1))
        return context + gate * delta + mixed


class ScreenModel(nn.Module):
    def __init__(self, kind: str, cfg: SinglePassConfig):
        super().__init__()
        self.kind = kind
        self.cfg = cfg
        self.pair_adapter = DeepResidualPairAdapter(cfg)
        self.text_projection = TextSearchProjection(cfg)
        if kind == "B1":
            self.temporal = FramewiseGatedDifferenceFusion(cfg.visual_dim)
        elif kind == "B2":
            self.temporal = TextConditionedTemporalFusion(cfg.visual_dim, cfg.text_dim)
        elif kind != "B0":
            raise ValueError(f"unsupported screen kind {kind}")
        self.logit_scale = nn.Parameter(torch.tensor(10.0).log())
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(
        self,
        visual: Tensor,
        base_text: Tensor,
        text_tokens: Tensor,
        attention: Tensor,
        content: Tensor,
    ) -> Tensor:
        if self.kind == "B0":
            native = visual
        elif self.kind == "B1":
            native = self.temporal(visual)
        else:
            native = self.temporal(visual, base_text)
        pair = self.pair_adapter(native).pair_search_vector
        text = self.text_projection(base_text, text_tokens, attention, content).text_search_vector
        return self.logit_scale.exp().clamp(1e-3, 100.0) * (text @ pair.T) + self.logit_bias

    def pair_embeddings(self, visual: Tensor, base_text: Tensor | None = None) -> Tensor:
        if self.kind == "B0":
            native = visual
        elif self.kind == "B1":
            native = self.temporal(visual)
        else:
            if base_text is None:
                raise ValueError("B2 pair embeddings require text")
            native = self.temporal(visual, base_text)
        return self.pair_adapter(native).pair_search_vector


@torch.no_grad()
def extract_visual_cache(
    rows: list[dict[str, Any]],
    mode: str,
    backend: UniverSatJointBackend,
    device: torch.device,
    batch_size: int,
    image_size: int,
    output_path: Path,
) -> dict[str, Any]:
    if mode == "joint":
        encoder = JointUniverSatSeriesEncoder(backend, output_grid=32, visual_dim=768).to(device).eval()
    else:
        frame = UniverSatFrameBackend(backend.model, output_grid=32, visual_dim=768).to(device).eval()
        encoder = SequenceUniverSatEncoder(frame, output_grid=32, visual_dim=768, freeze=True).to(device).eval()
    parts: list[Tensor] = []
    for start in range(0, len(rows), batch_size):
        indices = list(range(start, min(start + batch_size, len(rows))))
        batch = image_batch(rows, indices, image_size).to(device)
        result = encoder(batch)
        features = result.features if hasattr(result, "features") else result
        parts.append(features.detach().to(device="cpu", dtype=torch.float16))
    features = torch.cat(parts, dim=0)
    payload = {
        "schema_version": "qcpr-stage2-architecture-screen-feature-cache-v1",
        "mode": mode,
        "pair_ids": [row["pair_id"] for row in rows],
        "features": features,
        "feature_shape": list(features.shape),
        "image_size": image_size,
        "grid": [32, 32],
        "visual_dim": 768,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "mode": mode,
        "shape": list(features.shape),
        "pair_count": len(rows),
    }


def load_text_batch(text_encoder: nn.Module, captions: list[str], device: torch.device):
    with torch.no_grad():
        output = text_encoder(captions, role="query")
    return (
        output.global_embedding.to(device),
        output.token_embeddings.to(device),
        output.attention_mask.to(device),
        output.content_token_mask.to(device),
    )


def choose_caption(row: dict[str, Any], step: int, slot: int) -> str:
    captions = row["captions"]
    return str(captions[(step + slot) % len(captions)])


def make_batch_indices(pair_count: int, batch_size: int, step: int, seed: int) -> list[int]:
    generator = random.Random(seed + step * 1009)
    order = list(range(pair_count))
    generator.shuffle(order)
    start = (step * batch_size) % max(pair_count, 1)
    return order[start : start + batch_size] if start + batch_size <= pair_count else order[:batch_size]


def train_one(
    kind: str,
    train_rows: list[dict[str, Any]],
    visual_cache: Tensor,
    text_encoder: nn.Module,
    cfg: SinglePassConfig,
    device: torch.device,
    steps: int,
    microbatch: int,
    captions_per_pair: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = ScreenModel(kind, cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.05)
    losses = []
    peak_allocated = 0
    peak_reserved = 0
    started = time.monotonic()
    for step in range(steps):
        indices = make_batch_indices(len(train_rows), microbatch, step, seed)
        visual = visual_cache[indices].to(device, dtype=torch.float32)
        captions = [
            choose_caption(train_rows[index], step, slot)
            for index in indices
            for slot in range(captions_per_pair)
        ]
        if captions_per_pair > 1:
            visual_for_text = visual.repeat_interleave(captions_per_pair, dim=0)
        else:
            visual_for_text = visual
        base, tokens, attention, content = load_text_batch(text_encoder, captions, device)
        logits = model(visual_for_text, base, tokens, attention, content)
        pair_count = visual.shape[0]
        mapping = torch.arange(pair_count, device=device).repeat_interleave(captions_per_pair)
        positive = F.one_hot(mapping, num_classes=pair_count).bool()
        valid_negative = ~positive
        loss, stats = balanced_siglip_loss(logits, positive, valid_negative)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{kind} non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [parameter.grad for parameter in model.trainable_parameters() if parameter.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise RuntimeError(f"{kind} invalid gradients at step {step}")
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
    checkpoint = output_dir / f"{kind}_screen_checkpoint.pt"
    torch.save({"kind": kind, "model": model.state_dict(), "config": cfg.__dict__, "steps": steps, "seed": seed}, checkpoint)
    return {
        "kind": kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "steps": steps,
        "seed": seed,
        "microbatch": microbatch,
        "captions_per_pair": captions_per_pair,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "peak_allocated_gib": peak_allocated / (1024 ** 3),
        "peak_reserved_gib": peak_reserved / (1024 ** 3),
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in model.trainable_parameters()),
        "losses": losses,
    }


@torch.no_grad()
def evaluate_one(
    model: ScreenModel,
    rows: list[dict[str, Any]],
    visual_cache: Tensor,
    text_encoder: nn.Module,
    device: torch.device,
    text_batch: int,
) -> dict[str, Any]:
    model.eval()
    pair_ids = [row["pair_id"] for row in rows]
    pair_parts = []
    # B2 is query-conditioned; for gallery embeddings use a neutral zero query
    # only to expose its non-query-independent contract. Its official B2 score
    # below is evaluated query-conditioned, not indexed as A0.
    if model.kind == "B2":
        zero = torch.zeros(len(rows), 512, device=device)
        pair_parts.append(model.pair_embeddings(visual_cache.to(device, dtype=torch.float32), zero).cpu())
    else:
        for start in range(0, len(rows), 64):
            visual = visual_cache[start : start + 64].to(device, dtype=torch.float32)
            pair_parts.append(model.pair_embeddings(visual).cpu())
    pair_embeddings = torch.cat(pair_parts).to(device)
    captions = []
    mapping = []
    for pair_index, row in enumerate(rows):
        for caption in row["captions"]:
            captions.append(caption)
            mapping.append(pair_index)
    score_parts = []
    for start in range(0, len(captions), text_batch):
        chunk = captions[start : start + text_batch]
        base, tokens, attention, content = load_text_batch(text_encoder, chunk, device)
        if model.kind == "B2":
            visual = visual_cache[mapping[start : start + len(chunk)]].to(device, dtype=torch.float32)
        else:
            visual = visual_cache.to(device, dtype=torch.float32)
        if model.kind == "B2":
            logits = model(visual, base, tokens, attention, content)
        else:
            pair = pair_embeddings
            text = model.text_projection(base, tokens, attention, content).text_search_vector
            logits = model.logit_scale.exp().clamp(1e-3, 100.0) * (text @ pair.T) + model.logit_bias
        score_parts.append(logits.cpu())
    if model.kind == "B2":
        # query-conditioned scores are one score per query against its own item;
        # evaluate a full candidate matrix explicitly to avoid pretending B2 is
        # indexable. This path is expensive, so it is only used for a bounded
        # screen subset.
        all_scores = []
        for start in range(0, len(captions), text_batch):
            chunk = captions[start : start + text_batch]
            base, tokens, attention, content = load_text_batch(text_encoder, chunk, device)
            candidates = []
            for pair_start in range(0, len(rows), 64):
                v = visual_cache[pair_start : pair_start + 64].to(device, dtype=torch.float32)
                expanded = v.unsqueeze(0).expand(len(chunk), -1, -1, -1).reshape(-1, v.shape[1], v.shape[2])
                qbase = base[:, None].expand(-1, v.shape[0], -1).reshape(-1, base.shape[-1])
                # B2 fusion needs the query for every candidate.
                fused = model.temporal(expanded, qbase)
                pair = model.pair_adapter(fused).pair_search_vector
                qtext = model.text_projection(base, tokens, attention, content).text_search_vector
                pair_view = pair.reshape(len(chunk), v.shape[0], -1)
                logits = model.logit_scale.exp().clamp(1e-3, 100.0) * torch.bmm(qtext[:, None, :], pair_view.transpose(1, 2)).squeeze(1) + model.logit_bias
                candidates.append(logits)
            all_scores.append(torch.cat(candidates, dim=1).cpu())
        scores = torch.cat(all_scores, dim=0)
    else:
        scores = torch.cat(score_parts, dim=0)
    ranks = []
    for query_index, pair_index in enumerate(mapping):
        order = torch.argsort(scores[query_index], descending=True)
        rank = int((order == pair_index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks, dtype=torch.float64)
    return {
        "pair_count": len(rows),
        "query_count": len(captions),
        "metrics": {
            "r1": float((ranks_t <= 1).double().mean()),
            "r5": float((ranks_t <= 5).double().mean()),
            "r10": float((ranks_t <= 10).double().mean()),
            "r50": float((ranks_t <= 50).double().mean()),
            "r100": float((ranks_t <= 100).double().mean()),
            "mrr": float((1.0 / ranks_t).mean()),
            "mean_rank": float(ranks_t.mean()),
            "median_rank": float(ranks_t.median()),
        },
        "ranks": ranks,
        "pair_ids": pair_ids,
        "caption_to_pair": mapping,
        "score_contract": "query_conditioned_full_matrix" if model.kind == "B2" else "indexable_global_pair_embedding",
    }


def self_test() -> None:
    cfg = SinglePassConfig(visual_dim=12, text_dim=8, retrieval_dim=8, grid_size=4, pair_heads=4, dropout=0.0)
    for kind in ("B0", "B1", "B2"):
        model = ScreenModel(kind, cfg)
        frames = torch.randn(3, 2, 16, 12)
        base = torch.randn(3, 8)
        tokens = torch.randn(3, 5, 8)
        attention = torch.ones(3, 5, dtype=torch.bool)
        content = torch.ones(3, 5, dtype=torch.bool)
        if kind == "B0":
            visual = frames[:, 0]
        else:
            visual = frames
        scores = model(visual, base, tokens, attention, content)
        assert scores.shape == (3, 3)
        loss = scores.square().mean()
        loss.backward()
    print("architecture screen self-test passed")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--current-checkpoint", type=Path)
    p.add_argument("--train-manifest", type=Path)
    p.add_argument("--development-manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--extract-batch-size", type=int, default=8)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--captions-per-pair", type=int, default=2)
    p.add_argument("--max-train-pairs", type=int, default=2048)
    p.add_argument("--max-development-pairs", type=int, default=512)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    for required in (args.current_checkpoint, args.train_manifest, args.development_manifest, args.output_dir):
        if required is None:
            raise SystemExit("full screen requires --current-checkpoint, --train-manifest, --development-manifest and --output-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("compatible architecture screen requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    payload = torch.load(args.current_checkpoint, map_location="cpu", weights_only=False)
    backbone = payload["backbone_config"]
    train_rows = group_rows(read_rows(args.train_manifest), args.max_train_pairs)
    dev_rows = group_rows(read_rows(args.development_manifest), args.max_development_pairs)
    backend = UniverSatJointBackend(
        UniverSatBackendConfig(
            source_dir=backbone["universat_source"],
            checkpoint_dir=backbone["universat_checkpoint"],
            output_grid=32,
            freeze=True,
        )
    ).to(device).eval()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = output / "feature_cache"
    joint_meta = extract_visual_cache(train_rows + dev_rows, "joint", backend, device, args.extract_batch_size, args.image_size, cache_dir / "joint.pt")
    frame_meta = extract_visual_cache(train_rows + dev_rows, "framewise", backend, device, args.extract_batch_size, args.image_size, cache_dir / "framewise.pt")
    joint = torch.load(cache_dir / "joint.pt", map_location="cpu", weights_only=False)["features"]
    frame = torch.load(cache_dir / "framewise.pt", map_location="cpu", weights_only=False)["features"]
    text_encoder = JinaV5TextEncoder(JinaV5TextConfig(model_path=backbone["jina_model"], max_length=256, global_projection_mode="matryoshka_truncate", freeze=True)).to(device).eval()
    cfg = SinglePassConfig(grid_size=32, dropout=0.1)
    screen_rows = []
    for kind, cache in (("B0", joint), ("B1", frame), ("B2", frame)):
        train_meta = train_one(kind, train_rows, cache[: len(train_rows)], text_encoder, cfg, device, args.steps, args.microbatch, args.captions_per_pair, args.seed, output)
        ckpt = torch.load(train_meta["checkpoint"], map_location=device, weights_only=False)
        model = ScreenModel(kind, cfg).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        eval_rows = evaluate_one(model, dev_rows, cache[len(train_rows):], text_encoder, device, 64)
        screen_rows.append({"training": train_meta, "evaluation": eval_rows})
        del model
        torch.cuda.empty_cache()
    report = {
        "schema_version": "qcpr-stage2-compatible-architecture-screen-v1",
        "status": "PASS",
        "code_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "current_checkpoint": str(args.current_checkpoint),
        "current_checkpoint_sha256": sha256_file(args.current_checkpoint),
        "backbone": {
            "universat_source": backbone["universat_source"],
            "universat_checkpoint": backbone["universat_checkpoint"],
            "jina_model": backbone["jina_model"],
            "frozen": True,
        },
        "common_contract": {
            "train_pairs": len(train_rows),
            "development_pairs": len(dev_rows),
            "steps": args.steps,
            "microbatch": args.microbatch,
            "captions_per_pair": args.captions_per_pair,
            "seed": args.seed,
            "image_size": args.image_size,
            "grid": [32, 32],
            "train_schedule_sha256": stable_sha([row["pair_id"] for row in train_rows]),
            "development_schedule_sha256": stable_sha([row["pair_id"] for row in dev_rows]),
        },
        "feature_caches": {"joint": joint_meta, "framewise": frame_meta},
        "architectures": {item["training"]["kind"]: item for item in screen_rows},
        "scientific_notes": [
            "This is a bounded head-adaptation screen, not unrestricted P2 training.",
            "UniverSat and Jina parameters are frozen; only temporal/head/projection parameters are trained.",
            "B2 is query-conditioned and is not a precomputable A0 gallery index; its full candidate matrix is reported separately.",
            "Metrics are exact single-positive diagnostics on the common screen gallery; semantic claims require the verified semantic view.",
        ],
    }
    (output / "compatible_architecture_screen_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
