#!/usr/bin/env python3
"""Real native-resolution NaFlex encoding stress without text supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.chunking import build_synchronized_chunk_plan
from qcpr_siglip2.data.resolution import effective_resolution_record
from qcpr_siglip2.data.runtime import (
    encode_large_scene_images,
    encode_real_images,
    file_sha256,
)
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


def _read_highres_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("HIGHRES_RUNTIME_STRESS_READY") is not True:
        raise ValueError("high-resolution runtime manifest is not authorized")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("high-resolution runtime manifest has no items")
    return items


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _adapter_state(model: Siglip2TemporalRetrievalModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }


def _pair_embeddings(
    model: Siglip2TemporalRetrievalModel, image: Any
) -> torch.Tensor:
    temporal = model.temporal_adapter(
        image.patch_tokens,
        image.pooled_embedding,
        patch_valid_mask=image.patch_valid_mask,
        spatial_shapes=image.spatial_shapes,
        native_image_size=image.native_image_size,
        processed_patch_grid=image.processed_patch_grid,
        transform_hash=image.transform_hash,
        token_coordinates=image.token_coordinates,
        force_region_reduction=image.force_region_reduction,
    )
    return temporal.pair_cls.float().cpu()


def _write_nonsquare_pair(output: Path, size: tuple[int, int]) -> list[str]:
    width, height = size
    horizontal = Image.linear_gradient("L").resize((width, height))
    vertical = horizontal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    paths = []
    for index, channels in enumerate(
        ((horizontal, vertical, horizontal), (vertical, horizontal, vertical))
    ):
        path = output / f"engineering_nonsquare_t{index + 1}.png"
        Image.merge("RGB", channels).save(path)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--physical-registry")
    source.add_argument("--highres-stress-manifest")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--source", default="s2looking")
    parser.add_argument("--pair-count", type=int, default=16)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 576, 1024])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--overview-patch-budget", type=int, default=256)
    parser.add_argument("--engineering-2048", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = _git_state()
    if state != {"head": args.expected_code_sha, "worktree_clean": True}:
        raise RuntimeError("runtime code state does not match expected clean SHA")
    release = Path(args.data_release)
    registry = Path(args.physical_registry) if args.physical_registry else None
    stress_manifest = (
        Path(args.highres_stress_manifest) if args.highres_stress_manifest else None
    )
    checkpoint = Path(args.checkpoint)
    config = _load_config(Path(args.config_path))
    if registry is not None:
        source_rows = _read_jsonl(registry)
    else:
        if stress_manifest is None:
            raise RuntimeError("high-resolution source was not resolved")
        source_rows = _read_highres_manifest(stress_manifest)
    candidates = [
        row
        for row in source_rows
        if row.get("source") == args.source
        and row.get("item_type", "pair") == "pair"
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
            path = Path(str(frame.get("native_path", frame.get("path"))))
            expected_sha = str(frame.get("native_sha256", frame.get("sha256")))
            if file_sha256(path) != expected_sha:
                raise RuntimeError("native frame hash mismatch")
            height = int(frame.get("native_height_px", frame.get("height")))
            width = int(frame.get("native_width_px", frame.get("width")))
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
            embeddings = _pair_embeddings(model, image)
        valid_counts = image.patch_valid_mask.sum(-1).cpu()
        embeddings_by_budget[budget] = embeddings
        budget_results[str(budget)] = {
            "visual_token_shape": list(image.patch_tokens.shape),
            "processed_patch_grid": image.processed_patch_grid.cpu().tolist(),
            "valid_patch_count_min": int(valid_counts.min()),
            "valid_patch_count_max": int(valid_counts.max()),
            "embedding_shape": list(embeddings.shape),
            "embedding_norm_mean": float(embeddings.norm(dim=-1).mean()),
            "effective_rank": _effective_rank(embeddings),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del image
        torch.cuda.empty_cache()
    largest = max(args.budgets)
    if budget_results[str(largest)]["valid_patch_count_min"] < largest:
        raise RuntimeError("largest native budget did not produce the requested valid tokens")
    baseline = min(args.budgets)
    with torch.no_grad():
        repeat_image = encode_real_images(
            backbone, processor, pair_rows, device, max_num_patches=largest
        )
        repeated = _pair_embeddings(model, repeat_image)
    largest_repeat_difference = float(
        (embeddings_by_budget[largest] - repeated).abs().max()
    )
    if largest_repeat_difference > 1e-6:
        raise RuntimeError("query-independent pair vector is not deterministic")
    del repeat_image, repeated

    direct_resolution_records: list[dict[str, Any]] = []
    for budget in args.budgets:
        budget_record = budget_results[str(budget)]
        for pair_index, dimensions in enumerate(native_dimensions):
            grid = budget_record["processed_patch_grid"][pair_index][0]
            valid_count = int(grid[0]) * int(grid[1])
            direct_resolution_records.append(
                {
                    "item_id": pair_rows[pair_index]["canonical_pair_id"],
                    "patch_budget_name": f"NAFLEX_{budget}_PATCH_BUDGET",
                    **effective_resolution_record(
                        native_height=dimensions[0][0],
                        native_width=dimensions[0][1],
                        patch_size=config.patch_size,
                        actual_valid_patch_count=valid_count,
                        max_num_patches=budget,
                        processing_mode="DIRECT_NAFLEX",
                        processed_patch_grid=(int(grid[0]), int(grid[1])),
                    ),
                }
            )

    hierarchical_pair = pair_rows[:1]
    hierarchical_embeddings: dict[int, torch.Tensor] = {}
    hierarchical_results: dict[str, Any] = {}
    hierarchical_image = None
    for tile_batch_size in args.tile_batch_sizes:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            current_image = encode_large_scene_images(
                backbone,
                processor,
                hierarchical_pair,
                device,
                chunk_size=(args.tile_size, args.tile_size),
                overlap=(args.tile_overlap, args.tile_overlap),
                max_num_patches=largest,
                tile_batch_size=tile_batch_size,
                overview_max_num_patches=args.overview_patch_budget,
            )
            current_embedding = _pair_embeddings(model, current_image)
        hierarchical_embeddings[tile_batch_size] = current_embedding
        hierarchical_results[str(tile_batch_size)] = {
            "processing_mode": current_image.processing_mode,
            "visual_token_shape": list(current_image.patch_tokens.shape),
            "embedding_shape": list(current_embedding.shape),
            "chunk_counts": list(current_image.chunk_counts or ()),
            "per_tile_valid_patches": (
                list(current_image.chunk_valid_patch_counts[0])
                if current_image.chunk_valid_patch_counts is not None
                else None
            ),
            "chunk_size": list(current_image.chunk_size or ()),
            "chunk_overlap": list(current_image.chunk_overlap or ()),
            "overview_processed_patch_grid": (
                current_image.overview_processed_patch_grid.cpu().tolist()
                if current_image.overview_processed_patch_grid is not None
                else None
            ),
            "valid_patch_count_per_frame": int(
                current_image.patch_valid_mask[0, 0].sum()
            ),
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        if hierarchical_image is None:
            hierarchical_image = current_image
        else:
            del current_image
        torch.cuda.empty_cache()
    first_tile_batch = args.tile_batch_sizes[0]
    tile_batch_max_difference = max(
        float(
            (
                hierarchical_embeddings[first_tile_batch] - candidate
            ).abs().max()
        )
        for candidate in hierarchical_embeddings.values()
    )
    if tile_batch_max_difference > 1e-4:
        raise RuntimeError("tile batching changed the hierarchical pair embedding")
    if hierarchical_image is None:
        raise AssertionError("hierarchical image was not retained")
    native_height, native_width = native_dimensions[0][0]
    native_plan = build_synchronized_chunk_plan(
        (native_height, native_width),
        chunk_size=(args.tile_size, args.tile_size),
        overlap=(args.tile_overlap, args.tile_overlap),
    )
    overview_grid_values = hierarchical_image.overview_processed_patch_grid[0, 0]
    hierarchical_resolution_record = {
        "item_id": hierarchical_pair[0]["canonical_pair_id"],
        "tile_coordinates": [
            {
                "index": chunk.chunk_index,
                "top": chunk.top,
                "left": chunk.left,
                "bottom": chunk.bottom,
                "right": chunk.right,
            }
            for chunk in native_plan.chunks
        ],
        **effective_resolution_record(
            native_height=native_height,
            native_width=native_width,
            patch_size=config.patch_size,
            actual_valid_patch_count=int(
                hierarchical_image.patch_valid_mask[0, 0].sum()
            ),
            max_num_patches=largest,
            processing_mode="HIERARCHICAL_NATIVE",
            chunk_plan=native_plan,
            overview_patch_grid=(
                int(overview_grid_values[0]), int(overview_grid_values[1])
            ),
        ),
    }

    engineering_sizes = [(1536, 768)]
    if args.engineering_2048:
        engineering_sizes.append((2048, 2048))
    engineering_results: list[dict[str, Any]] = []
    for width, height in engineering_sizes:
        paths = _write_nonsquare_pair(output, (width, height))
        plan = build_synchronized_chunk_plan(
            (height, width),
            chunk_size=(args.tile_size, args.tile_size),
            overlap=(args.tile_overlap, args.tile_overlap),
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            engineering_image = encode_large_scene_images(
                backbone,
                processor,
                [{"frames": paths}],
                device,
                chunk_size=(args.tile_size, args.tile_size),
                overlap=(args.tile_overlap, args.tile_overlap),
                max_num_patches=largest,
                tile_batch_size=max(args.tile_batch_sizes),
                overview_max_num_patches=args.overview_patch_budget,
            )
            engineering_embedding = _pair_embeddings(model, engineering_image)
        engineering_results.append(
            {
                "scientific_evidence": False,
                "engineering_only": True,
                "native_width": width,
                "native_height": height,
                "tile_count": len(plan.chunks),
                "coverage_fraction": effective_resolution_record(
                    native_height=height,
                    native_width=width,
                    patch_size=config.patch_size,
                    actual_valid_patch_count=int(
                        engineering_image.patch_valid_mask[0, 0].sum()
                    ),
                    max_num_patches=largest,
                    processing_mode="HIERARCHICAL_NATIVE",
                    chunk_plan=plan,
                )["native_pixel_coverage_fraction"],
                "embedding_shape": list(engineering_embedding.shape),
                "seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
        )
        del engineering_image, engineering_embedding
        torch.cuda.empty_cache()

    roundtrip_path = output / "highres_adapter_roundtrip.pt"
    torch.save({"model_state": _adapter_state(model)}, roundtrip_path)
    fresh_model = Siglip2TemporalRetrievalModel(backbone, config).to(device).eval()
    roundtrip_sha = _restore_adapter(fresh_model, roundtrip_path)
    with torch.no_grad():
        reload_image = encode_real_images(
            backbone, processor, pair_rows, device, max_num_patches=largest
        )
        reloaded = _pair_embeddings(fresh_model, reload_image)
        hierarchical_reloaded = _pair_embeddings(fresh_model, hierarchical_image)
    checkpoint_max_difference = float(
        (embeddings_by_budget[largest] - reloaded).abs().max()
    )
    if checkpoint_max_difference > 1e-6:
        raise RuntimeError("high-resolution adapter checkpoint roundtrip mismatch")
    hierarchical_checkpoint_max_difference = float(
        (
            hierarchical_embeddings[first_tile_batch] - hierarchical_reloaded
        ).abs().max()
    )
    if hierarchical_checkpoint_max_difference > 1e-6:
        raise RuntimeError("hierarchical adapter checkpoint roundtrip mismatch")
    del reload_image, reloaded, hierarchical_reloaded, fresh_model, hierarchical_image
    cross_budget_cosine = {
        str(budget): float(
            torch.nn.functional.cosine_similarity(
                embeddings_by_budget[baseline], embeddings, dim=-1
            ).mean()
        )
        for budget, embeddings in embeddings_by_budget.items()
    }
    result = {
        "status": "HIGHRES_INPUT_RUNTIME_PASS",
        "evaluation_only": True,
        "training_enabled": False,
        "retrieval_metrics_computed": False,
        "retrieval_blocker": "verified high-resolution text view is not authorized",
        "code": state,
        "release": str(release),
        "release_sha256sums_sha256": file_sha256(release / "SHA256SUMS"),
        "physical_registry": str(registry) if registry is not None else None,
        "physical_registry_sha256": file_sha256(registry) if registry else None,
        "highres_stress_manifest": (
            str(stress_manifest) if stress_manifest is not None else None
        ),
        "highres_stress_manifest_sha256": (
            file_sha256(stress_manifest) if stress_manifest is not None else None
        ),
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
        "direct_mode_interpretation": (
            "DIRECT_NAFLEX accepts native 1024px inputs but the 1024-patch "
            "budget processes an approximately 512px square grid; it is not "
            "FULL_NATIVE_1024_DETAIL_PASS."
        ),
        "hierarchical_native": {
            "status": "FULL_NATIVE_LARGE_SCENE_RUNTIME_PASS",
            "results_by_tile_batch_size": hierarchical_results,
            "tile_batch_max_absolute_difference": tile_batch_max_difference,
            "aggregator_latent_count": config.large_scene_latents,
            "ann_vectors_per_physical_item": 1,
            "region_vectors_exposed_to_ann": False,
            "resolution": hierarchical_resolution_record,
        },
        "engineering_stress": engineering_results,
        "cross_budget_cosine_vs_smallest": cross_budget_cosine,
        "one_vector_determinism": {
            "status": "PASS",
            "largest_budget": largest,
            "embedding_sha256": _tensor_sha256(embeddings_by_budget[largest]),
            "max_absolute_difference": largest_repeat_difference,
        },
        "checkpoint_roundtrip": {
            "status": "PASS",
            "checkpoint": str(roundtrip_path),
            "checkpoint_sha256": roundtrip_sha,
            "max_absolute_difference": checkpoint_max_difference,
            "hierarchical_max_absolute_difference": (
                hierarchical_checkpoint_max_difference
            ),
        },
    }
    report = output / "native_highres_stress.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "native_highres_stress.sha256").write_text(
        f"{file_sha256(report)}  {report.name}\n"
    )
    resolution_report = output / "reports" / "effective_resolution_audit.json"
    resolution_report.parent.mkdir(parents=True, exist_ok=True)
    resolution_report.write_text(
        json.dumps(
            {
                "schema_version": "qcpr-effective-resolution-v1",
                "direct_records": direct_resolution_records,
                "hierarchical_native_record": hierarchical_resolution_record,
                "engineering_records": engineering_results,
                "direct_full_native_1024_detail": False,
                "hierarchical_full_native_1024_detail": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"NATIVE_HIGHRES_STRESS_FAILED: {exc}", file=sys.stderr)
        raise
