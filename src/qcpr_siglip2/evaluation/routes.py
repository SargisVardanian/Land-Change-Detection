"""Deterministic direct-versus-hierarchical route audits."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from ..backbones.siglip2 import Siglip2Backbone
from ..data.runtime import encode_large_scene_images, encode_real_images, encode_real_text
from ..evaluation.retrieval import first_positive_ranks, full_gallery_metrics
from .common_gallery import exact_relevance_masks, global_stage_scores


def _distribution(values: Tensor) -> dict[str, Any]:
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p10": float(torch.quantile(values, 0.10)),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _encode_pair_route(
    model: Any,
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    max_num_patches: int,
    hierarchical_chunk_size: tuple[int, int],
) -> Tensor:
    embeddings: list[Tensor] = []
    for start in range(0, len(pair_rows), batch_size):
        batch = pair_rows[start : start + batch_size]
        if mode == "DIRECT_NAFLEX":
            image = encode_real_images(
                backbone,
                processor,
                batch,
                device,
                max_num_patches=max_num_patches,
                no_grad=True,
            )
        elif mode == "HIERARCHICAL_NATIVE":
            image = encode_large_scene_images(
                backbone,
                processor,
                batch,
                device,
                chunk_size=hierarchical_chunk_size,
                overlap=(0, 0),
                max_num_patches=max_num_patches,
                tile_batch_size=4,
                overview_max_num_patches=max_num_patches,
                no_grad=True,
            )
        else:
            raise ValueError(f"unsupported route audit mode: {mode}")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
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
        embeddings.append(torch.nn.functional.normalize(temporal.pair_cls.float(), dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


def route_audit(
    model: Any,
    backbone: Siglip2Backbone,
    processor: Any,
    query_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    pair_count: int = 64,
    pair_batch_size: int = 4,
    query_batch_size: int = 64,
    max_num_patches: int = 256,
    hierarchical_chunk_size: tuple[int, int] = (128, 128),
) -> dict[str, Any]:
    """Encode one fixed dev subset through both routes and compare ranks."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    selected_pairs = list(pair_rows[: min(pair_count, len(pair_rows))])
    selected_ids = {str(row["canonical_pair_id"]) for row in selected_pairs}
    selected_queries = [
        row
        for row in query_rows
        if str(row.get("canonical_pair_id")) in selected_ids
    ]
    if not selected_queries:
        raise ValueError("route audit subset has no queries")
    positive, ignored = exact_relevance_masks(selected_queries, selected_pairs)
    query_embeddings: list[Tensor] = []
    for start in range(0, len(selected_queries), query_batch_size):
        text = encode_real_text(
            backbone,
            processor,
            selected_queries[start : start + query_batch_size],
            device,
            no_grad=True,
        )
        query_embeddings.append(
            torch.nn.functional.normalize(text.pooled_embedding.float(), dim=-1).cpu()
        )
    text_embeddings = torch.cat(query_embeddings, dim=0)
    route_embeddings = {
        mode: _encode_pair_route(
            model,
            backbone,
            processor,
            selected_pairs,
            device,
            mode=mode,
            batch_size=pair_batch_size,
            max_num_patches=max_num_patches,
            hierarchical_chunk_size=hierarchical_chunk_size,
        )
        for mode in ("DIRECT_NAFLEX", "HIERARCHICAL_NATIVE")
    }
    scores = {
        mode: global_stage_scores(text_embeddings, embedding)
        for mode, embedding in route_embeddings.items()
    }
    ranks = {
        mode: first_positive_ranks(value, positive)
        for mode, value in scores.items()
    }
    cosine = torch.nn.functional.cosine_similarity(
        route_embeddings["DIRECT_NAFLEX"],
        route_embeddings["HIERARCHICAL_NATIVE"],
        dim=-1,
    )
    rank_difference = ranks["DIRECT_NAFLEX"].float() - ranks[
        "HIERARCHICAL_NATIVE"
    ].float()
    return {
        "status": "PASS",
        "subset": {
            "pair_count": len(selected_pairs),
            "query_count": len(selected_queries),
            "ordered_pair_ids": [str(row["canonical_pair_id"]) for row in selected_pairs],
            "ordered_query_ids": [
                str(row.get("caption_id", row.get("query_id")))
                for row in selected_queries
            ],
        },
        "route_contract": {
            "direct": "DIRECT_NAFLEX",
            "hierarchical": "HIERARCHICAL_NATIVE",
            "max_num_patches": max_num_patches,
            "hierarchical_chunk_size": list(hierarchical_chunk_size),
            "same_query_rows": True,
            "same_relevance_masks": True,
        },
        "direct": {
            "metrics": full_gallery_metrics(scores["DIRECT_NAFLEX"], positive),
            "embedding_shape": list(route_embeddings["DIRECT_NAFLEX"].shape),
        },
        "hierarchical": {
            "metrics": full_gallery_metrics(
                scores["HIERARCHICAL_NATIVE"], positive
            ),
            "embedding_shape": list(route_embeddings["HIERARCHICAL_NATIVE"].shape),
        },
        "same_pair_embedding_cosine": _distribution(cosine),
        "rank_difference_direct_minus_hierarchical": _distribution(rank_difference),
        "rank_difference_absolute": _distribution(rank_difference.abs()),
        "ignored_cells": int(ignored.sum()),
    }


__all__ = ["route_audit"]
