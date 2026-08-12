#!/usr/bin/env python3
"""Real UniSat/Jina integration smoke for QCPR v3; no synthetic inputs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from qcpr_v3.config import config_dict, default_config
from qcpr_v3.data.contracts import TemporalMetadata, assert_mask_free_record
from qcpr_v3.diagnostics.evidence import evidence_gradient_diagnostics
from qcpr_v3.evaluation.retrieval import rank_scores, retrieval_metrics
from qcpr_v3.models import QCPRV3Model
from qcpr_v3.models.jina_query_encoder import JinaQueryEncoder, QueryFeatures
from qcpr_v3.models.temporal_adapter import TemporalOutput
from qcpr_v3.training.exposure import ExposureAccounting, record_step
from qcpr_v3.training.objective import UnifiedListwiseLoss

DIRECTION_WORDS = {
    "appeared", "appear", "built", "constructed", "added", "increased",
    "expanded", "disappeared", "disappear", "removed", "demolished",
    "decreased", "reduced",
}


@dataclass(frozen=True)
class Sample:
    pair_id: str
    query_id: str
    text: str
    t1: Path
    t2: Path
    source: str
    row: dict[str, Any]


class FrozenJinaBase(nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenJinaBase:
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            [text if text.startswith("Query: ") else f"Query: {text}" for text in texts],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.model(**encoded, return_dict=True)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != 1024:
            raise RuntimeError(f"Jina output must be [B,L,1024], got {tuple(hidden.shape)}")
        return hidden, encoded["attention_mask"].bool()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def check_code(expected: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    head = git_value(root, "rev-parse", "HEAD")
    clean = not bool(git_value(root, "status", "--porcelain"))
    if head != expected:
        raise RuntimeError(f"runtime HEAD mismatch: {head} != {expected}")
    if not clean:
        raise RuntimeError("runtime worktree is dirty")
    return {
        "expected_sha": expected,
        "local_sha": head,
        "branch": git_value(root, "branch", "--show-current"),
        "worktree_clean": clean,
        "parent_sha": git_value(root, "rev-parse", "HEAD^"),
        "commit_timestamp": git_value(root, "show", "-s", "--format=%cI", "HEAD"),
    }


def has_direction(row: dict[str, Any]) -> bool:
    words = set(str(row.get("caption", "")).casefold().replace("-", " ").split())
    return bool(words & DIRECTION_WORDS)


def choose_samples(
    manifest: Path, physical_batch: int, captions_per_pair: int
) -> tuple[list[Sample], list[str], list[str], int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(manifest):
        assert_mask_free_record(row)
        if str(row.get("dataset_name")) not in {"levir_mci", "second_cc"}:
            continue
        if str(row.get("query_scope")) != "exact_pair":
            continue
        if row.get("is_generated") or row.get("verification") == "generated_unverified":
            continue
        text = str(row.get("caption", row.get("text", ""))).strip()
        pair_id = str(row.get("canonical_pair_id", row.get("pair_id", "")))
        t1, t2 = Path(str(row.get("t1_path", ""))), Path(str(row.get("t2_path", "")))
        if not pair_id or not text or not t1.is_file() or not t2.is_file():
            raise FileNotFoundError(f"invalid real row {pair_id}: {t1} {t2}")
        groups[pair_id].append(row)
    usable: list[tuple[str, list[dict[str, Any]]]] = []
    for pair_id, rows in groups.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            unique.setdefault(str(row.get("caption", row.get("text", ""))).strip(), row)
        rows = sorted(unique.values(), key=lambda row: (
            not has_direction(row), str(row.get("caption_id", ""))
        ))
        if len(rows) >= captions_per_pair:
            usable.append((pair_id, rows))
    if len(usable) < physical_batch:
        raise RuntimeError(f"only {len(usable)} usable pairs, need {physical_batch}")
    by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for item in usable:
        by_source[str(item[1][0].get("dataset_name"))].append(item)
    for rows in by_source.values():
        rows.sort(key=lambda item: item[0])
    selected: list[tuple[str, list[dict[str, Any]]]] = []
    sources = sorted(by_source)
    cursor = 0
    while len(selected) < physical_batch:
        added = False
        for source in sources:
            if cursor < len(by_source[source]):
                selected.append(by_source[source][cursor])
                added = True
                if len(selected) == physical_batch:
                    break
        if not added:
            break
        cursor += 1
    if len(selected) != physical_batch:
        raise RuntimeError("source-balanced real batch could not be filled")
    direction_pair = next(
        (i for i, (_, rows) in enumerate(selected)
         if any(has_direction(row) for row in rows[:captions_per_pair])),
        -1,
    )
    if direction_pair > 0:
        selected[0], selected[direction_pair] = selected[direction_pair], selected[0]
    samples: list[Sample] = []
    pair_ids: list[str] = []
    query_ids: list[str] = []
    direction_query = -1
    for pair_index, (pair_id, rows) in enumerate(selected):
        pair_ids.append(pair_id)
        for caption_index, row in enumerate(rows[:captions_per_pair]):
            query_id = str(row.get("caption_id", f"{pair_id}:smoke:{caption_index}"))
            samples.append(Sample(
                pair_id, query_id,
                str(row.get("caption", row.get("text", ""))).strip(),
                Path(str(row["t1_path"])), Path(str(row["t2_path"])),
                str(row.get("dataset_name")), row,
            ))
            query_ids.append(query_id)
            if direction_query < 0 and has_direction(row):
                direction_query = pair_index * captions_per_pair + caption_index
    if len(set(pair_ids)) != physical_batch or len(set(query_ids)) != len(query_ids):
        raise RuntimeError("real smoke ID contract failed")
    return samples, pair_ids, query_ids, direction_query


def unique_pairs(samples: list[Sample]) -> list[Sample]:
    result: list[Sample] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.pair_id not in seen:
            result.append(sample)
            seen.add(sample.pair_id)
    return result


def load_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.load()
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def load_frames(samples: list[Sample], device: torch.device) -> tuple[Tensor, float]:
    start = time.perf_counter()
    pairs = [torch.stack((load_rgb(s.t1), load_rgb(s.t2))) for s in unique_pairs(samples)]
    return torch.stack(pairs).to(device), time.perf_counter() - start


def make_coordinates(token_count: int, batch: int, device: torch.device) -> Tensor:
    grid = int(round(token_count ** 0.5))
    if grid * grid != token_count:
        raise RuntimeError(f"native token count is not a square grid: {token_count}")
    axis = torch.linspace(0.0, 1.0, grid, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, token_count, 2).expand(batch, -1, -1)


def make_metadata(batch: int, device: torch.device, reverse: bool = False) -> TemporalMetadata:
    values = torch.tensor(
        [[1.0, 0.0] if reverse else [0.0, 1.0]], device=device
    ).expand(batch, -1)
    frames = torch.tensor(
        [[1, 0] if reverse else [0, 1]], device=device, dtype=torch.long
    ).expand(batch, -1)
    missing = torch.ones((batch, 2), dtype=torch.bool, device=device)
    return TemporalMetadata(values, values - values[:, :1], frames, metadata_missing=missing)


def amp(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def build_components(
    source: Path, unisat_checkpoint: Path, jina_path: Path,
    initial_checkpoint: Path, device: torch.device
) -> tuple[nn.Module, nn.Module, FrozenJinaBase, QCPRV3Model, dict[str, Any]]:
    from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
    from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
    from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
    loader = UniverSatJointBackend(UniverSatBackendConfig(
        source_dir=source, checkpoint_dir=unisat_checkpoint,
        output_grid=32, visual_dim=768, freeze=True,
    )).to(device).eval()
    frame = UniverSatFrameBackend(loader.model, output_grid=32, visual_dim=768).to(device).eval()
    visual = SequenceUniverSatEncoder(frame, output_grid=32, visual_dim=768, freeze=True).to(device).eval()
    jina_base = FrozenJinaBase(jina_path).to(device).eval()
    config = default_config()
    text_encoder = JinaQueryEncoder(
        input_dim=1024, hidden_dim=512, adapter_layers=config.text.adapter_layers,
        heads=config.text.heads, ff_ratio=config.text.ff_ratio,
        dropout=config.text.dropout, base_encoder=jina_base,
    )
    model = QCPRV3Model(config, text_encoder=text_encoder).to(device)
    model.freeze_backbones()
    payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("initial checkpoint has no model state")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith("text_encoder.base_encoder.")]
    if unexpected:
        raise RuntimeError(f"unexpected initial checkpoint keys: {unexpected[:8]}")
    return visual, loader, jina_base, model, {
        "initial_missing_keys": len(missing),
        "initial_unexpected_keys": len(unexpected),
        "unisat_identifier": "g-astruc/UniverSat",
        "unisat_source_revision": git_value(source, "rev-parse", "HEAD"),
        "jina_identifier": "jinaai/jina-embeddings-v5-text-small-retrieval",
    }


def parameter_groups(
    visual: nn.Module, jina_base: nn.Module, model: QCPRV3Model
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    projection = model.temporal_adapter.input_projection
    temporal = [
        (name, p) for name, p in model.temporal_adapter.named_parameters()
        if not name.startswith("input_projection.") and name != "change_seed"
    ]
    text_projection: list[tuple[str, nn.Parameter]] = []
    for prefix, module in (
        ("input_projection", model.text_encoder.input_projection),
        ("output_projection", model.text_encoder.output_projection),
    ):
        text_projection.extend((f"{prefix}.{n}", p) for n, p in module.named_parameters())
    text_adapter: list[tuple[str, nn.Parameter]] = []
    for prefix, module in (
        ("adapter", model.text_encoder.adapter),
        ("adapter_norm", model.text_encoder.adapter_norm),
    ):
        text_adapter.extend((f"{prefix}.{n}", p) for n, p in module.named_parameters())
    text_adapter.append(("adapter_residual_scale", model.text_encoder.adapter_residual_scale))
    return {
        "unisat_backbone": list(visual.named_parameters()),
        "jina_base_encoder": list(jina_base.named_parameters()),
        "unisat_projection": list(projection.named_parameters()),
        "temporal_adapter": temporal,
        "jina_projection": text_projection,
        "jina_text_adapter": text_adapter,
        "evidence_bottleneck": list(model.evidence_bottleneck.named_parameters()),
        "change_tokens": [("change_seed", model.temporal_adapter.change_seed)],
        "relevance_model": list(model.relevance_model.named_parameters()),
    }


def group_contract(groups: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, Any]:
    return {
        name: {
            "parameter_count": sum(p.numel() for _, p in entries),
            "trainable_count": sum(p.numel() for _, p in entries if p.requires_grad),
            "requires_grad": sorted({bool(p.requires_grad) for _, p in entries}),
            "parameter_names": [n for n, _ in entries],
        }
        for name, entries in groups.items()
    }


def group_gradients(groups: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, entries in groups.items():
        params = [p for _, p in entries]
        grads = [p.grad.detach() for p in params if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads]) if grads else None
        finite = None if flat is None else torch.isfinite(flat)
        finite_values = None if finite is None else flat[finite]
        result[name] = {
            "parameter_count": sum(p.numel() for p in params),
            "trainable_count": sum(p.numel() for p in params if p.requires_grad),
            "requires_grad": sorted({bool(p.requires_grad) for p in params}),
            "parameters_with_grad": sum(p.grad is not None for p in params),
            "gradient_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0,
            "gradient_min": None if finite_values is None or finite_values.numel() == 0 else float(finite_values.min()),
            "gradient_max": None if finite_values is None or finite_values.numel() == 0 else float(finite_values.max()),
            "finite_gradient_fraction": float(finite.float().mean()) if finite is not None else 1.0,
        }
    return result


def relevance(samples: list[Sample], pair_ids: list[str], device: torch.device) -> tuple[Tensor, Tensor, bool]:
    index = {item: i for i, item in enumerate(pair_ids)}
    grades = torch.zeros((len(samples), len(pair_ids)), dtype=torch.long, device=device)
    valid = torch.ones_like(grades, dtype=torch.bool)
    for qi, sample in enumerate(samples):
        positives = [str(v) for v in sample.row.get("positive_pair_ids", [sample.pair_id])]
        matched = [v for v in positives if v in index]
        if not matched:
            raise RuntimeError(f"query has no selected positive: {sample.query_id}")
        for item in matched:
            grades[qi, index[item]] = 1
        for item in [str(v) for v in sample.row.get("ignored_pair_ids", [])]:
            if item in index:
                if grades[qi, index[item]] > 0:
                    raise RuntimeError(f"positive and ignored overlap: {sample.query_id}")
                valid[qi, index[item]] = False
    return grades, valid, bool((grades > 0).sum(dim=1).max() > 1)


def encode_batch(
    frames: Tensor, visual: nn.Module, model: QCPRV3Model, texts: list[str],
    coords: Tensor, metadata: TemporalMetadata, grades: Tensor, valid: Tensor,
    objective: UnifiedListwiseLoss, device: torch.device
) -> tuple[Any, dict[str, float]]:
    start = time.perf_counter()
    with amp(device):
        encoded = visual(frames)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    uni_time = time.perf_counter() - start
    native = encoded.features.detach()
    if tuple(native.shape[1:]) != (2, 1024, 768):
        raise RuntimeError(f"native shape mismatch: {tuple(native.shape)}")
    start = time.perf_counter()
    with amp(device):
        visual_out = model.encode_visual(native, coords, metadata)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    temporal_time = time.perf_counter() - start
    start = time.perf_counter()
    with amp(device):
        query = model.encode_query(texts)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    jina_time = time.perf_counter() - start
    start = time.perf_counter()
    with amp(device):
        scored = model.score(query, visual_out)
        loss = objective(scored.scores, grades, valid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (scored, loss, query, visual_out, native, encoded.features), {
        "unisat_feature_seconds": uni_time,
        "temporal_adapter_seconds": temporal_time,
        "jina_encoding_seconds": jina_time,
        "score_and_loss_seconds": time.perf_counter() - start,
    }


def select_visual(visual: TemporalOutput, indices: list[int]) -> TemporalOutput:
    index = torch.tensor(indices, device=visual.sequence_cls.device)
    t = visual.temporal
    temporal = TemporalMetadata(
        t.timestamps.index_select(0, index), t.delta_times.index_select(0, index),
        t.frame_ids.index_select(0, index),
        None if t.sensor_ids is None else t.sensor_ids.index_select(0, index),
        None if t.gsd is None else t.gsd.index_select(0, index),
        None if t.metadata_missing is None else t.metadata_missing.index_select(0, index),
    )
    return replace(
        visual,
        sequence_cls=visual.sequence_cls.index_select(0, index),
        frame_cls=visual.frame_cls.index_select(0, index),
        change_tokens=visual.change_tokens.index_select(0, index),
        dense_tokens=visual.dense_tokens.index_select(0, index),
        coordinates=visual.coordinates.index_select(0, index),
        frame_mask=visual.frame_mask.index_select(0, index),
        token_mask=visual.token_mask.index_select(0, index),
        temporal=temporal,
    )


def select_query(query: QueryFeatures, indices: list[int]) -> QueryFeatures:
    index = torch.tensor(indices, device=query.text_cls.device)
    return replace(
        query,
        text_cls=query.text_cls.index_select(0, index),
        text_tokens=query.text_tokens.index_select(0, index),
        text_mask=query.text_mask.index_select(0, index),
    )


def zero_evidence_score(model: QCPRV3Model, query: QueryFeatures, visual: TemporalOutput) -> Tensor:
    text = torch.nn.functional.normalize(query.text_cls, dim=-1)
    sequence = torch.nn.functional.normalize(visual.sequence_cls, dim=-1)
    global_score = text @ sequence.transpose(0, 1)
    change = torch.nn.functional.normalize(visual.change_tokens, dim=-1)
    slot = torch.logsumexp(torch.einsum("qd,pkd->qpk", text, change), dim=-1)
    slot -= torch.log(torch.tensor(float(change.shape[1]), device=text.device, dtype=text.dtype))
    zero = torch.zeros(
        query.text_cls.shape[0], visual.sequence_cls.shape[0], model.config.retrieval_dim,
        device=text.device, dtype=text.dtype
    )
    return model.relevance_model(
        global_score, torch.zeros_like(global_score), slot,
        torch.zeros_like(global_score), text, sequence, zero
    ).score


def remove_flat_token_indices(token_mask: Tensor, indices: Tensor) -> Tensor:
    """Return a token mask with flattened [T,N] token indices disabled."""
    if token_mask.ndim != 3 or token_mask.shape[0] < 1:
        raise ValueError("token_mask must be [B,T,N] with a non-empty batch")
    if indices.ndim != 1:
        raise ValueError("indices must be a flat [M] tensor")
    result = token_mask.clone()
    time_count, tokens_per_frame = result.shape[1], result.shape[2]
    for value in indices.tolist():
        frame, token = divmod(int(value), tokens_per_frame)
        if frame >= time_count:
            raise IndexError(f"token index {value} is outside [T={time_count},N={tokens_per_frame}]")
        result[0, frame, token] = False
    if not bool(result.any()):
        result[0, 0, 0] = True
    return result


def evidence_probe(
    model: QCPRV3Model, native: Tensor, coords: Tensor,
    metadata: TemporalMetadata, texts: list[str], device: torch.device,
    detach: bool
) -> float:
    model.zero_grad(set_to_none=True)
    model.train()
    with amp(device):
        visual = model.encode_visual(native, coords, metadata)
        query = model.encode_query(texts)
        scored = model.score(query, visual)
        if scored.evidence is None:
            raise RuntimeError("evidence is disabled")
        vector = scored.evidence.evidence_vector.detach() if detach else scored.evidence.evidence_vector
        text = torch.nn.functional.normalize(query.text_cls, dim=-1)
        vector = torch.nn.functional.normalize(vector, dim=-1)
        (text[:, None, :] * vector).sum().backward()
    grad = model.evidence_bottleneck.query_projection.weight.grad
    return 0.0 if grad is None else float(grad.detach().float().norm())


def evidence_diagnostics(
    model: QCPRV3Model, visual: TemporalOutput, query: QueryFeatures,
    samples: list[Sample], direction_query: int, frames: Tensor,
    visual_encoder: nn.Module, coords: Tensor, metadata: TemporalMetadata,
    native: Tensor, device: torch.device
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        with amp(device):
            pair_visual = select_visual(visual, [0])
            pair_query = select_query(query, [0, 1])
            swapped = model.score(pair_query, pair_visual)
            maps = swapped.evidence.relevance_weights[:, 0]
            first, second = maps[0].reshape(-1).float(), maps[1].reshape(-1).float()
            normal = float(swapped.scores[0, 0])
            zero = float(zero_evidence_score(model, select_query(query, [0]), pair_visual)[0, 0])
            count = max(1, first.numel() // 10)
            top = torch.topk(first, count).indices
            bottom = torch.argsort(first)[:count]

            def remove(indices: Tensor) -> TemporalOutput:
                return replace(
                    pair_visual,
                    token_mask=remove_flat_token_indices(pair_visual.token_mask, indices),
                )

            top_score = float(model.score(select_query(query, [0]), remove(top)).scores[0, 0])
            bottom_score = float(model.score(select_query(query, [0]), remove(bottom)).scores[0, 0])
            if direction_query < 0:
                raise RuntimeError("TEMPORAL_DIRECTION_NOT_EXERCISED")
            pair_count = len({sample.pair_id for sample in samples})
            captions = len(samples) // pair_count
            pair_index = direction_query // captions
            reverse_frames = frames[pair_index:pair_index + 1].flip(1)
            reverse_native = visual_encoder(reverse_frames).features.detach()
            reverse_visual = model.encode_visual(
                reverse_native, coords[:1], make_metadata(1, device, reverse=True)
            )
            original = float(model.score(select_query(query, [direction_query]), select_visual(visual, [pair_index])).scores[0, 0])
            reversed_score = float(model.score(select_query(query, [direction_query]), reverse_visual).scores[0, 0])
    normal_grad = evidence_probe(model, native, coords, metadata, [s.text for s in samples], device, False)
    detached_grad = evidence_probe(model, native, coords, metadata, [s.text for s in samples], device, True)
    return {
        "query_swap_l1": float((first - second).abs().mean()),
        "query_swap_cosine": float(torch.nn.functional.cosine_similarity(first, second, dim=0)),
        "score_change_after_query_swap": abs(float(swapped.scores[0, 0] - swapped.scores[1, 0])),
        "score_with_zero_evidence": zero,
        "score_change_after_zero_evidence": abs(normal - zero),
        "score_drop_top_evidence": normal - top_score,
        "score_drop_bottom_evidence": normal - bottom_score,
        "top_evidence_count": count,
        "weights_shape": list(swapped.evidence.relevance_weights.shape),
        "directional_query_id": samples[direction_query].query_id,
        "time_reversal_original_score": original,
        "time_reversal_score": reversed_score,
        "time_reversal_score_delta": abs(original - reversed_score),
        "evidence_path_gradient_norm": normal_grad,
        "detached_evidence_path_gradient_norm": detached_grad,
    }


def load_state(model: QCPRV3Model, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("checkpoint has no model state")
    _, unexpected = model.load_state_dict(payload["model"], strict=False)
    bad = [key for key in unexpected if not key.startswith("text_encoder.base_encoder.")]
    if bad:
        raise RuntimeError(f"unexpected checkpoint keys: {bad[:8]}")


def child(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    state = check_code(args.expected_code_sha)
    samples, pair_ids, query_ids, _ = choose_samples(
        args.train_manifest, args.physical_batch_size, args.captions_per_pair
    )
    visual, _, _, model, _ = build_components(
        args.unisat_source, args.unisat_checkpoint, args.jina_model,
        args.checkpoint_path, device
    )
    load_state(model, args.output_dir / "checkpoint.pt")
    model.eval()
    frames, _ = load_frames(samples, device)
    with torch.no_grad():
        with amp(device):
            encoded = visual(frames)
            native = encoded.features.detach()
            coords = make_coordinates(native.shape[2], len(pair_ids), device)
            output = model.score(
                model.encode_query([s.text for s in samples]),
                model.encode_visual(native, coords, make_metadata(len(pair_ids), device)),
            )
    write_json(args.output_dir / "roundtrip_child_scores.json", {
        "code_state": state, "pair_ids": pair_ids, "query_ids": query_ids,
        "scores": output.scores.float().cpu().tolist(),
    })


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.steps <= 32:
        raise SystemExit("steps must be in [1,32]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/H100 is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    random.seed(20260804)
    np.random.seed(20260804)
    torch.manual_seed(20260804)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.manual_seed_all(20260804)
    state = check_code(args.expected_code_sha)
    train_sha, dev_sha = sha256_file(args.train_manifest), sha256_file(args.development_manifest)
    release_meta = args.data_release / "dataset_v2_stage2_release.json"
    if not args.data_release.is_dir() or not release_meta.is_file():
        raise FileNotFoundError(f"invalid release: {args.data_release}")
    release_sha = sha256_file(release_meta)
    for path in (
        args.checkpoint_path,
        args.unisat_checkpoint / "model.safetensors",
        args.jina_model / "model.safetensors",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    samples, pair_ids, query_ids, direction_query = choose_samples(
        args.train_manifest, args.physical_batch_size, args.captions_per_pair
    )
    grades, valid, multi_positive = relevance(samples, pair_ids, device)
    visual, loader, jina_base, model, load_info = build_components(
        args.unisat_source, args.unisat_checkpoint, args.jina_model,
        args.checkpoint_path, device
    )
    groups = parameter_groups(visual, jina_base, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.config_path is not None and not args.config_path.is_file():
        raise FileNotFoundError(args.config_path)
    write_json(args.output_dir / "parameter_groups.json", {
        "phase": "PRE_FIRST_STEP", "groups": group_contract(groups)
    })
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)
    objective = UnifiedListwiseLoss(temperature=default_config().training.temperature)
    exposure = ExposureAccounting([], [], [], [])
    history: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    smoke_start = time.perf_counter()
    final: tuple[Any, ...] | None = None
    final_frames: Tensor | None = None
    final_coords: Tensor | None = None
    final_metadata: TemporalMetadata | None = None
    for step in range(args.steps):
        record_step(exposure, pair_ids, query_ids)
        optimizer.zero_grad(set_to_none=True)
        frames, decode_time = load_frames(samples, device)
        coords = make_coordinates(1024, len(pair_ids), device)
        metadata = make_metadata(len(pair_ids), device)
        forward_start = time.perf_counter()
        batch, timing = encode_batch(
            frames, visual, model, [s.text for s in samples],
            coords, metadata, grades, valid, objective, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        scored, loss, query, visual_out, native, raw_native = batch
        backward_start = time.perf_counter()
        loss.loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_start
        if not torch.isfinite(loss.loss):
            raise FloatingPointError(f"nonfinite loss at step {step + 1}")
        grads = [p.grad for p in trainable if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
        clip = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer_start = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_step_seconds = time.perf_counter() - optimizer_start
        timings.append({
            "image_decode_seconds": decode_time,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_step_seconds": optimizer_step_seconds,
            **timing,
        })
        history.append({
            "step": step + 1, "loss": float(loss.loss.detach().float()),
            "positive_count": loss.positive_count,
            "valid_query_count": loss.valid_query_count,
            "mean_positive_score": float(loss.mean_positive_score.detach().float()),
            "mean_all_score": float(loss.mean_all_score.detach().float()),
            "gradient_norm_before_clip": clip,
            "evidence_gradient_norm": float(
                model.evidence_bottleneck.query_projection.weight.grad.detach().float().norm()
            ) if model.evidence_bottleneck.query_projection.weight.grad is not None else 0.0,
        })
        final, final_frames, final_coords, final_metadata = batch, frames, coords, metadata
    if final is None or final_frames is None or final_coords is None or final_metadata is None:
        raise RuntimeError("no final batch")
    final_gradients = group_gradients(groups)
    for name, values in final_gradients.items():
        if values["trainable_count"] and not values["parameters_with_grad"]:
            raise RuntimeError(f"TRAINABLE_MODULE_NO_GRADIENT:{name}")
        if values["requires_grad"] == [False] and values["parameters_with_grad"]:
            raise RuntimeError(f"FROZEN_BACKBONE_HAS_GRADIENT:{name}")
        if values["finite_gradient_fraction"] < 1.0:
            raise RuntimeError(f"NONFINITE_GRADIENT:{name}")
    model.eval()
    with torch.no_grad():
        frames_eval, _ = load_frames(samples, device)
        with amp(device):
            encoded = visual(frames_eval)
            native_eval = encoded.features.detach()
            coords_eval = make_coordinates(native_eval.shape[2], len(pair_ids), device)
            metadata_eval = make_metadata(len(pair_ids), device)
            visual_eval = model.encode_visual(native_eval, coords_eval, metadata_eval)
            query_eval = model.encode_query([s.text for s in samples])
            scored_eval = model.score(query_eval, visual_eval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unisat_sha = sha256_file(args.unisat_checkpoint / "model.safetensors")
    jina_sha = sha256_file(args.jina_model / "model.safetensors")
    checkpoint = args.output_dir / "checkpoint.pt"
    meta = {
        "code_sha": args.expected_code_sha, "release_sha256": release_sha,
        "train_manifest_sha256": train_sha, "development_manifest_sha256": dev_sha,
        "unisat_checkpoint_sha256": unisat_sha, "jina_checkpoint_sha256": jina_sha,
        "global_step": args.steps, "requested_steps": args.steps,
    }
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": None, "rng_state": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        },
        "step": args.steps, "metadata": meta,
    }, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    (args.output_dir / "checkpoint.sha256").write_text(checkpoint_sha + "\n")
    reference = scored_eval.scores.float().cpu().tolist()
    write_json(args.output_dir / "roundtrip_reference.json", {"scores": reference})
    child_command = [
        sys.executable, str(Path(__file__).resolve()), "--roundtrip-child",
        "--data-release", str(args.data_release),
        "--train-manifest", str(args.train_manifest),
        "--development-manifest", str(args.development_manifest),
        "--unisat-source", str(args.unisat_source),
        "--unisat-checkpoint", str(args.unisat_checkpoint),
        "--jina-model", str(args.jina_model),
        "--expected-code-sha", args.expected_code_sha,
        "--output-dir", str(args.output_dir),
        "--checkpoint-path", str(args.checkpoint_path),
        "--steps", "0", "--physical-batch-size", str(args.physical_batch_size),
        "--captions-per-pair", str(args.captions_per_pair),
    ]
    child_result = subprocess.run(child_command, check=False)
    if child_result.returncode:
        raise RuntimeError(f"checkpoint child failed: {child_result.returncode}")
    child_scores = np.asarray(json.loads(
        (args.output_dir / "roundtrip_child_scores.json").read_text()
    )["scores"], dtype=np.float32)
    before_scores = np.asarray(reference, dtype=np.float32)
    max_abs = float(np.max(np.abs(before_scores - child_scores)))
    max_rel = float(np.max(
        np.abs(before_scores - child_scores) / np.maximum(np.abs(before_scores), 1e-6)
    ))
    roundtrip = bool(np.allclose(before_scores, child_scores, atol=1e-4, rtol=1e-4))
    write_json(args.output_dir / "checkpoint_roundtrip.json", {
        "passed": roundtrip, "fresh_process": True,
        "max_abs_score_difference": max_abs, "max_relative_score_difference": max_rel,
        "tolerance": {"atol": 1e-4, "rtol": 1e-4},
    })
    if not roundtrip:
        raise RuntimeError("CHECKPOINT_ROUNDTRIP_MISMATCH")
    evidence = evidence_diagnostics(
        model, visual_eval, query_eval, samples, direction_query,
        frames_eval, visual, coords_eval, metadata_eval, native_eval, device
    )
    failures: list[str] = []
    if evidence["query_swap_l1"] <= 1e-6 and evidence["query_swap_cosine"] >= 0.999999:
        failures.append("EVIDENCE_NOT_QUERY_CONDITIONED")
    if evidence["score_change_after_zero_evidence"] <= 1e-8:
        failures.append("EVIDENCE_NOT_CAUSAL")
    if evidence["score_drop_top_evidence"] <= evidence["score_drop_bottom_evidence"]:
        failures.append("EVIDENCE_NOT_CAUSAL")
    if evidence["evidence_path_gradient_norm"] <= 1e-8 or evidence["detached_evidence_path_gradient_norm"] > 1e-8:
        failures.append("EVIDENCE_NOT_CAUSAL")
    if evidence["time_reversal_score_delta"] <= 1e-6:
        failures.append("TEMPORAL_DIRECTION_NOT_LEARNABLE")

    write_json(args.output_dir / "code_state.json", {
        **state, "data_release": str(args.data_release),
        "data_release_metadata_sha256": release_sha,
        "train_manifest_sha256": train_sha, "development_manifest_sha256": dev_sha,
        "initial_checkpoint": str(args.checkpoint_path),
        "initial_checkpoint_sha256": sha256_file(args.checkpoint_path),
        "unisat_checkpoint_sha256": unisat_sha, "jina_checkpoint_sha256": jina_sha,
        "preprocessing_sha256": stable_sha({
            "resize": [256, 256], "channels": "RGB", "range": "0..1",
            "interpolation": "BILINEAR",
        }),
    })
    write_json(args.output_dir / "config_resolved.json", {
        "config_path": None if args.config_path is None else str(args.config_path),
        "config_sha256": None if args.config_path is None else sha256_file(args.config_path),
        "config": config_dict(default_config()), "steps": args.steps,
        "physical_batch": len(pair_ids), "captions_per_pair": args.captions_per_pair,
    })
    write_json(args.output_dir / "environment.json", {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "bf16_autocast": True, "matmul_precision": "high",
    })
    write_json(args.output_dir / "model_contract.json", {
        "architecture_id": default_config().architecture_id,
        "unisat": {
            "identifier": "g-astruc/UniverSat",
            "source_revision": git_value(args.unisat_source, "rev-parse", "HEAD"),
            "checkpoint_sha256": unisat_sha, "native_shape": list(native_eval.shape),
        },
        "jina": {
            "identifier": "jinaai/jina-embeddings-v5-text-small-retrieval",
            "revision": json.loads((args.jina_model / "config.json").read_text())["transformers_version"],
            "checkpoint_sha256": jina_sha,
            "text_tokens": list(query_eval.text_tokens.shape),
            "text_cls": list(query_eval.text_cls.shape),
        },
        "outputs": {
            "sequence_cls": list(visual_eval.sequence_cls.shape),
            "frame_cls": list(visual_eval.frame_cls.shape),
            "change_tokens": list(visual_eval.change_tokens.shape),
            "dense_tokens": list(visual_eval.dense_tokens.shape),
            "score_matrix": list(scored_eval.scores.shape),
        },
        "mask_access": "none", "synthetic_inputs": False,
        "multi_positive_runtime": "EXERCISED" if multi_positive else "MULTI_POSITIVE_RUNTIME_NOT_EXERCISED",
        "initial_checkpoint_load": load_info,
    })
    write_json(args.output_dir / "parameter_groups.json", {
        "phase": "PRE_FIRST_STEP", "groups": group_contract(groups),
        "final_gradient_diagnostics": final_gradients,
    })
    write_json(args.output_dir / "batch_contract.json", {
        "physical_microbatch": len(pair_ids), "logical_physical_batch": len(pair_ids),
        "captions_per_pair": args.captions_per_pair, "query_count": len(query_ids),
        "gradient_accumulation": 1, "logical_score_matrix": [len(query_ids), len(pair_ids)],
        "steps": args.steps, "native_shape": list(native_eval.shape),
        "mask_access": "none",
        "multi_positive_runtime": "EXERCISED" if multi_positive else "MULTI_POSITIVE_RUNTIME_NOT_EXERCISED",
    })
    write_json(args.output_dir / "exposure_accounting.json", exposure.as_dict())
    (args.output_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in history) + "\n"
    )
    write_json(args.output_dir / "evaluation_metrics.json", retrieval_metrics(
        scored_eval.scores.float(), grades.bool(), ks=(1, 5, len(pair_ids))
    ))
    torch.save(scored_eval.scores.float().cpu(), args.output_dir / "full_rankings.pt")
    order = rank_scores(scored_eval.scores.float()).cpu().tolist()
    with (args.output_dir / "rankings_top100.jsonl").open("w") as handle:
        for query_id, row_order in zip(query_ids, order, strict=True):
            handle.write(json.dumps({
                "query_id": query_id,
                "ranked_item_ids": [pair_ids[i] for i in row_order[:100]],
            }) + "\n")
    ranking_sha = sha256_file(args.output_dir / "full_rankings.pt")
    write_json(args.output_dir / "retrieval_integrity_audit.json", {
        "evaluation_scope": "real_smoke_selected_train_batch",
        "ranking_shape": list(scored_eval.scores.shape), "ranking_sha256": ranking_sha,
        "ordered_pair_ids_sha256": sequence_sha(pair_ids),
        "ordered_query_ids_sha256": sequence_sha(query_ids),
        "train_manifest_sha256": train_sha, "development_manifest_sha256": dev_sha,
        "data_release_metadata_sha256": release_sha,
        "valid_mask_all_true": bool(valid.all()),
    })
    write_json(args.output_dir / "embedding_diagnostics.json", {
        "sequence_cls_norm_mean": float(visual_eval.sequence_cls.float().norm(dim=-1).mean()),
        "text_cls_norm_mean": float(query_eval.text_cls.float().norm(dim=-1).mean()),
        "dense_token_norm_mean": float(visual_eval.dense_tokens.float().norm(dim=-1).mean()),
    })
    write_json(args.output_dir / "gradient_diagnostics.json", {
        "final_step": final_gradients, "evidence": evidence_gradient_diagnostics(model),
    })
    write_json(args.output_dir / "evidence_diagnostics.json", evidence)
    write_json(args.output_dir / "localization_metrics.json", {
        "mask_access": "evaluation_only", "diagnostics": evidence,
    })
    write_json(args.output_dir / "runtime_profile.json", {
        "gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__, "step_timings": timings,
        "mean_timings": {
            key: float(np.mean([row[key] for row in timings])) for key in timings[0]
        },
        "total_wall_seconds": time.perf_counter() - smoke_start,
        "cpu_rss_peak_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "text_token_count": int(query_eval.text_tokens.shape[1]),
    })
    write_json(args.output_dir / "cuda_memory.json", {
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3),
        "current_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3),
        "current_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3),
        "native_visual_tokens": int(native_eval.shape[2]),
        "score_matrix": list(scored_eval.scores.shape),
    })
    write_json(args.output_dir / "training_complete.json", {
        "status": "COMPLETED_FIXED_STEPS", "global_step": args.steps,
        "requested_steps": args.steps, "checkpoint_sha256": checkpoint_sha,
        "diagnostic_failures": failures,
    })
    write_json(args.output_dir / "smoke_status.json", {
        "status": "PASS" if not failures else "FAIL",
        "model_v3_real_integration_smoke": "PASS" if not failures else "FAIL",
        "failures": failures, "no_mask_access": True, "synthetic_inputs": False,
    })
    sums = {
        path.name: sha256_file(path)
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (args.output_dir / "SHA256SUMS").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in sorted(sums.items())) + "\n"
    )
    if failures:
        raise RuntimeError(";".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-release", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--unisat-source", type=Path, required=True)
    parser.add_argument("--unisat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--roundtrip-child", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.roundtrip_child:
        child(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
