#!/usr/bin/env python3
"""Real native-resolution NaFlex encoding stress without text supervision."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.runtime import encode_real_images, file_sha256
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("registry rows must be JSON objects")
                rows.append(value)
    return rows


def _git_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return {"head": head, "worktree_clean": not dirty}


def _load_config(path: Path) -> Siglip2TemporalConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        name: payload[name]
        for name in Siglip2TemporalConfig.__dataclass_fields__
        if name in payload
    }
    return Siglip2TemporalConfig.from_dict(fields)


def _restore_adapter(model: Siglip2TemporalRetrievalModel, checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain model_state")
    adapter = {key: value for key, value in state.items() if not key.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise RuntimeError("adapter checkpoint mismatch")
    return file_sha256(checkpoint)


def _effective_rank(embeddings: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(embeddings - embeddings.mean(0, keepdim=True))
    mass = singular / singular.sum().clamp_min(1e-12)
    return float((-(mass * mass.clamp_min(1e-12).log()).sum()).exp())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--physical-registry", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--source", default="s2looking")
    parser.add_argument("--pair-count", type=int, default=16)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 576, 1024])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = _git_state()
    if state != {"head": args.expected_code_sha, "worktree_clean": True}:
        raise RuntimeError("runtime code state does not match expected clean SHA")
    release = Path(args.data_release)
    registry = Path(args.physical_registry)
    checkpoint = Path(args.checkpoint)
    config = _load_config(Path(args.config_path))
    candidates = [
        row
        for row in _read_jsonl(registry)
        if row.get("source") == args.source
        and row.get("item_type") == "pair"
        and len(row.get("frames", [])) == 2
    ]
    selected = sorted(candidates, key=lambda row: str(row["item_id"]))[
        : args.pair_count
    ]
    if len(selected) != args.pair_count:
        raise ValueError("insufficient native high-resolution pairs")
    largest_side_required = config.patch_size * math.isqrt(max(args.budgets))
    pair_rows: list[dict[str, Any]] = []
    native_dimensions: list[list[list[int]]] = []
    for row in selected:
        paths: list[str] = []
        dimensions: list[list[int]] = []
        for frame in row["frames"]:
            path = Path(str(frame["native_path"]))
            if file_sha256(path) != str(frame["native_sha256"]):
                raise RuntimeError("native frame hash mismatch")
            height = int(frame["native_height_px"])
            width = int(frame["native_width_px"])
            if min(height, width) < largest_side_required:
                raise RuntimeError("native image would require upsampling for target budget")
            paths.append(str(path))
            dimensions.append([height, width])
        pair_rows.append({"frames": paths, "canonical_pair_id": row["item_id"]})
        native_dimensions.append(dimensions)

    device = torch.device("cuda")
    backbone = Siglip2Backbone(
        args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    model = Siglip2TemporalRetrievalModel(backbone, config).to(device).eval()
    checkpoint_sha = _restore_adapter(model, checkpoint)
    embeddings_by_budget: dict[int, torch.Tensor] = {}
    budget_results: dict[str, Any] = {}
    for budget in args.budgets:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            image = encode_real_images(
                backbone,
                processor,
                pair_rows,
                device,
                max_num_patches=budget,
            )
            temporal = model.temporal_adapter(
                image.patch_tokens,
                image.pooled_embedding,
                patch_valid_mask=image.patch_valid_mask,
                spatial_shapes=image.spatial_shapes,
                native_image_size=image.native_image_size,
                processed_patch_grid=image.processed_patch_grid,
                transform_hash=image.transform_hash,
            )
            embeddings = temporal.pair_cls.float().cpu()
        valid_counts = image.patch_valid_mask.sum(-1).cpu()
        embeddings_by_budget[budget] = embeddings
        budget_results[str(budget)] = {
            "visual_token_shape": list(image.patch_tokens.shape),
            "valid_patch_count_min": int(valid_counts.min()),
            "valid_patch_count_max": int(valid_counts.max()),
            "embedding_shape": list(embeddings.shape),
            "embedding_norm_mean": float(embeddings.norm(dim=-1).mean()),
            "effective_rank": _effective_rank(embeddings),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del image, temporal
        torch.cuda.empty_cache()
    largest = max(args.budgets)
    if budget_results[str(largest)]["valid_patch_count_min"] < largest:
        raise RuntimeError("largest native budget did not produce the requested valid tokens")
    baseline = min(args.budgets)
    cross_budget_cosine = {
        str(budget): float(
            torch.nn.functional.cosine_similarity(
                embeddings_by_budget[baseline], embeddings, dim=-1
            ).mean()
        )
        for budget, embeddings in embeddings_by_budget.items()
    }
    result = {
        "status": "HIGHRES_ENCODING_PASS_RETRIEVAL_TEXT_BLOCKED",
        "evaluation_only": True,
        "training_enabled": False,
        "retrieval_metrics_computed": False,
        "retrieval_blocker": "verified high-resolution text view is not authorized",
        "code": state,
        "release": str(release),
        "release_sha256sums_sha256": file_sha256(release / "SHA256SUMS"),
        "physical_registry": str(registry),
        "physical_registry_sha256": file_sha256(registry),
        "checkpoint_sha256": checkpoint_sha,
        "source": args.source,
        "pair_count": len(selected),
        "pair_ids": [str(row["item_id"]) for row in selected],
        "native_dimensions": native_dimensions,
        "gsd_available": False,
        "footprint_available": False,
        "no_upsampling": True,
        "mask_access": False,
        "budgets": budget_results,
        "cross_budget_cosine_vs_smallest": cross_budget_cosine,
    }
    report = output / "native_highres_stress.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "native_highres_stress.sha256").write_text(
        f"{file_sha256(report)}  {report.name}\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"NATIVE_HIGHRES_STRESS_FAILED: {exc}", file=sys.stderr)
        raise
