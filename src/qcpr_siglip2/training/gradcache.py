"""Exact logical-batch feature caching for temporal SigLIP-2 training."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

import torch
from torch import Tensor

from ..data.runtime import RawFeatureBatch, _device_autocast, encode_real_features
from ..models.model import Siglip2TemporalRetrievalModel
from .objective import pair_balanced_symmetric_multi_positive_listwise_loss


@dataclass(frozen=True)
class CachedLogicalFeatures:
    frame_tokens: Tensor
    frame_embeddings: Tensor
    text_tokens: Tensor
    text_embeddings: Tensor
    text_mask: Tensor
    patch_valid_mask: Tensor | None = None
    spatial_shapes: Tensor | None = None
    native_image_size: Tensor | None = None
    processed_patch_grid: Tensor | None = None
    transform_hash: str | None = None
    token_coordinates: Tensor | None = None
    processing_mode: str = "DIRECT_NAFLEX"
    force_region_reduction: bool = False


def module_gradient_report(
    model: Siglip2TemporalRetrievalModel,
) -> dict[str, dict[str, int | float | bool]]:
    """Audit trainable and frozen gradient state by model component."""

    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "temporal_adapter": [],
        "evidence_bottleneck": [],
        "relevance_model": [],
        "retrieval_temperature": [],
        "siglip2_vision_backbone": [],
        "siglip2_text_backbone": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("temporal_adapter."):
            bucket = "temporal_adapter"
        elif name.startswith("evidence_bottleneck."):
            bucket = "evidence_bottleneck"
        elif name.startswith("relevance_model."):
            bucket = "relevance_model"
        elif name == "log_temperature":
            bucket = "retrieval_temperature"
        elif name.startswith(("backbone.vision_model.", "backbone.model.vision_model.")):
            bucket = "siglip2_vision_backbone"
        elif name.startswith(("backbone.text_model.", "backbone.model.text_model.")):
            bucket = "siglip2_text_backbone"
        else:
            continue
        buckets[bucket].append((name, parameter))

    report: dict[str, dict[str, int | float | bool]] = {}
    for bucket, named_parameters in buckets.items():
        parameters = [parameter for _, parameter in named_parameters]
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        gradients = [
            parameter.grad.detach().float()
            for parameter in trainable
            if parameter.grad is not None
        ]
        frozen_with_grad = [
            name
            for name, parameter in named_parameters
            if not parameter.requires_grad and parameter.grad is not None
        ]
        flat = (
            torch.cat([gradient.reshape(-1) for gradient in gradients])
            if gradients
            else torch.empty(0)
        )
        finite = bool(torch.isfinite(flat).all()) if flat.numel() else True
        report[bucket] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_count": sum(parameter.numel() for parameter in trainable),
            "parameters_with_grad": len(gradients),
            "gradient_norm": float(flat.norm()) if flat.numel() else 0.0,
            "gradient_min": float(flat.min()) if flat.numel() else 0.0,
            "gradient_max": float(flat.max()) if flat.numel() else 0.0,
            "finite_gradient_fraction": (
                float(torch.isfinite(flat).float().mean()) if flat.numel() else 1.0
            ),
            "finite": finite,
            "frozen_parameters_with_grad": len(frozen_with_grad),
        }
        if trainable and not gradients:
            raise RuntimeError(f"TRAINABLE_MODULE_NO_GRADIENT:{bucket}")
        if frozen_with_grad:
            raise RuntimeError(f"FROZEN_BACKBONE_HAS_GRADIENT:{bucket}")
        if not finite:
            raise FloatingPointError(f"NONFINITE_GRADIENT:{bucket}")
    return report


def _requires_grad(value: Tensor) -> Tensor:
    return value.detach().requires_grad_(True)


def cache_features(features: RawFeatureBatch) -> CachedLogicalFeatures:
    """Detach backbone outputs while retaining feature-gradient endpoints."""

    return CachedLogicalFeatures(
        frame_tokens=_requires_grad(features.frame_tokens),
        frame_embeddings=_requires_grad(features.frame_embeddings),
        text_tokens=_requires_grad(features.text_tokens),
        text_embeddings=_requires_grad(features.text_embeddings),
        text_mask=features.text_mask.detach(),
        patch_valid_mask=(
            features.patch_valid_mask.detach()
            if features.patch_valid_mask is not None
            else None
        ),
        spatial_shapes=(
            features.spatial_shapes.detach()
            if features.spatial_shapes is not None
            else None
        ),
        native_image_size=(
            features.native_image_size.detach()
            if features.native_image_size is not None
            else None
        ),
        processed_patch_grid=(
            features.processed_patch_grid.detach()
            if features.processed_patch_grid is not None
            else None
        ),
        transform_hash=features.transform_hash,
        token_coordinates=(
            features.token_coordinates.detach()
            if features.token_coordinates is not None
            else None
        ),
        processing_mode=features.processing_mode,
        force_region_reduction=features.force_region_reduction,
    )


def _cat_optional_tensor(
    chunks: list[RawFeatureBatch], attribute: str
) -> Tensor | None:
    values = [getattr(chunk, attribute) for chunk in chunks]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"inconsistent optional feature metadata: {attribute}")
    return torch.cat(cast(list[Tensor], values), dim=0)


def _query_offsets(
    query_pair_indices: Tensor | list[int] | tuple[int, ...] | None,
    *,
    pair_count: int,
    captions_per_pair: int | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Validate local query ownership and return indices plus chunk offsets."""

    if query_pair_indices is None:
        if captions_per_pair is None or captions_per_pair <= 0:
            raise ValueError(
                "variable-Q batches require query_pair_indices; legacy batches "
                "must provide a positive captions_per_pair"
            )
        indices = torch.arange(pair_count, device=device, dtype=torch.long).repeat_interleave(
            captions_per_pair
        )
    else:
        indices = torch.as_tensor(query_pair_indices, device=device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("query_pair_indices must be a non-empty one-dimensional sequence")
    if torch.any(indices < 0) or torch.any(indices >= pair_count):
        raise ValueError("query_pair_indices contains an invalid pair index")
    expected = torch.arange(pair_count, device=indices.device, dtype=torch.long)
    counts = torch.bincount(indices, minlength=pair_count)
    if not torch.equal(torch.nonzero(counts, as_tuple=False).flatten(), expected):
        raise ValueError("every physical pair must own at least one query")
    offsets = torch.cat(
        [
            torch.zeros(1, device=indices.device, dtype=torch.long),
            counts.cumsum(0),
        ]
    )
    if int(offsets[-1]) != int(indices.numel()):
        raise ValueError("query offsets do not cover all query rows")
    if indices.tolist() != sorted(indices.tolist()):
        raise ValueError("query rows must be grouped by physical pair for GradCache")
    return indices, offsets


def _feature_surrogate(
    features: RawFeatureBatch,
    feature_grads: CachedLogicalFeatures,
    pair_start: int,
    pair_end: int,
    query_start: int,
    query_end: int | None = None,
) -> Tensor:
    # Backward-compatible helper form used by the legacy unit tests.  The
    # active variable-Q path always passes explicit query_start/query_end.
    if query_end is None:
        captions_per_pair = query_start
        query_start = pair_start * captions_per_pair
        query_end = pair_end * captions_per_pair
    expected_pairs = pair_end - pair_start
    expected_queries = query_end - query_start
    if features.frame_tokens.shape[0] != expected_pairs:
        raise ValueError("recomputed frame features do not match pair slice")
    if features.text_tokens.shape[0] != expected_queries:
        raise ValueError("recomputed text features do not match query slice")
    return (
        (features.frame_tokens * feature_grads.frame_tokens[pair_start:pair_end]).sum()
        + (
            features.frame_embeddings
            * feature_grads.frame_embeddings[pair_start:pair_end]
        ).sum()
        + (
            features.text_tokens
            * feature_grads.text_tokens[query_start:query_end]
        ).sum()
        + (
            features.text_embeddings
            * feature_grads.text_embeddings[query_start:query_end]
        ).sum()
    )


def _encode_logical_features_in_chunks(
    backbone: Any,
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    device: torch.device,
    physical_batch_size: int,
    query_offsets: Tensor | list[int] | tuple[int, ...] | None = None,
    captions_per_pair: int | None = None,
    dtype: torch.dtype,
    no_grad: bool,
    max_num_patches: int | None = None,
    is_naflex: bool | None = None,
    representation_mode: str = "DIRECT_NAFLEX",
    hierarchical_chunk_size: tuple[int, int] = (128, 128),
    hierarchical_tile_batch_size: int = 4,
    hierarchical_overview_max_num_patches: int | None = None,
    max_text_length: int = 64,
) -> RawFeatureBatch:
    """Encode one logical batch through bounded physical microbatches.

    The listwise objective is still evaluated once after concatenation.  The
    chunking only bounds frozen-backbone activation and decoder input memory;
    it must never turn the logical batch into independent losses.
    """

    if len(pair_rows) % physical_batch_size:
        raise ValueError("logical pair count must divide physical microbatch size")
    if query_offsets is None:
        _, offsets = _query_offsets(
            None,
            pair_count=len(pair_rows),
            captions_per_pair=captions_per_pair,
        )
    else:
        offsets = torch.as_tensor(query_offsets, dtype=torch.long)
    if offsets.ndim != 1 or offsets.numel() != len(pair_rows) + 1:
        raise ValueError("query_offsets must have pair_count + 1 entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(query_rows):
        raise ValueError("query_offsets do not cover query rows")
    if torch.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("every pair must have at least one unique query")
    chunks: list[RawFeatureBatch] = []
    for start in range(0, len(pair_rows), physical_batch_size):
        end = start + physical_batch_size
        query_start = int(offsets[start])
        query_end = int(offsets[end])
        chunks.append(
            encode_real_features(
                backbone,
                processor,
                pair_rows[start:end],
                query_rows[query_start:query_end],
                device,
                dtype=dtype,
                no_grad=no_grad,
                max_num_patches=max_num_patches,
                is_naflex=is_naflex,
                representation_mode=representation_mode,
                hierarchical_chunk_size=hierarchical_chunk_size,
                hierarchical_tile_batch_size=hierarchical_tile_batch_size,
                hierarchical_overview_max_num_patches=hierarchical_overview_max_num_patches,
                max_text_length=max_text_length,
            )
        )
    if not chunks:
        raise ValueError("logical batch must contain at least one pair")
    modes = {chunk.processing_mode for chunk in chunks}
    reductions = {chunk.force_region_reduction for chunk in chunks}
    if len(modes) != 1 or len(reductions) != 1:
        raise ValueError("logical batch mixes representation modes")
    transform_hashes = [chunk.transform_hash for chunk in chunks]
    if all(value is None for value in transform_hashes):
        transform_hash = None
    elif any(value is None for value in transform_hashes):
        raise ValueError("inconsistent optional feature metadata: transform_hash")
    else:
        transform_hash = sha256(
            "\n".join(cast(list[str], transform_hashes)).encode("utf-8")
        ).hexdigest()
    return RawFeatureBatch(
        frame_tokens=torch.cat([chunk.frame_tokens for chunk in chunks], dim=0),
        frame_embeddings=torch.cat(
            [chunk.frame_embeddings for chunk in chunks], dim=0
        ),
        text_tokens=torch.cat([chunk.text_tokens for chunk in chunks], dim=0),
        text_embeddings=torch.cat(
            [chunk.text_embeddings for chunk in chunks], dim=0
        ),
        text_mask=torch.cat([chunk.text_mask for chunk in chunks], dim=0),
        patch_valid_mask=_cat_optional_tensor(chunks, "patch_valid_mask"),
        spatial_shapes=_cat_optional_tensor(chunks, "spatial_shapes"),
        native_image_size=_cat_optional_tensor(chunks, "native_image_size"),
        processed_patch_grid=_cat_optional_tensor(chunks, "processed_patch_grid"),
        transform_hash=transform_hash,
        token_coordinates=_cat_optional_tensor(chunks, "token_coordinates"),
        processing_mode=chunks[0].processing_mode,
        force_region_reduction=chunks[0].force_region_reduction,
    )


def logical_listwise_step(
    model: Siglip2TemporalRetrievalModel,
    backbone: Any,
    processor: Any,
    pair_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    query_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    positive_mask: Tensor,
    ignored_mask: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    physical_batch_size: int,
    query_pair_indices: Tensor | list[int] | tuple[int, ...] | None = None,
    captions_per_pair: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    dtype: torch.dtype = torch.bfloat16,
    gradient_clip_norm: float = 1.0,
    recompute_backbone: bool = False,
    max_num_patches: int | None = None,
    is_naflex: bool | None = None,
    representation_mode: str = "DIRECT_NAFLEX",
    hierarchical_chunk_size: tuple[int, int] = (128, 128),
    hierarchical_tile_batch_size: int = 4,
    hierarchical_overview_max_num_patches: int | None = None,
    max_text_length: int = 64,
) -> dict[str, Any]:
    """Run one common logical score matrix and exact feature-gradient replay.

    The listwise loss is evaluated once over the complete logical batch.  When
    the top pretrained blocks are trainable, feature gradients are replayed
    through physical microbatches (GradCache-style); no independent
    microbatch loss is substituted for the logical objective.
    """

    if len(pair_rows) % physical_batch_size:
        raise ValueError("logical pair count must divide physical microbatch size")
    query_indices, query_offsets = _query_offsets(
        query_pair_indices,
        pair_count=len(pair_rows),
        captions_per_pair=captions_per_pair,
        device=device,
    )
    if len(query_rows) != int(query_indices.numel()):
        raise ValueError("query rows do not match variable-Q query ownership")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    frozen_features = _encode_logical_features_in_chunks(
        backbone,
        processor,
        pair_rows,
        query_rows,
        device=device,
        physical_batch_size=physical_batch_size,
        query_offsets=query_offsets,
        dtype=dtype,
        no_grad=True,
        max_num_patches=max_num_patches,
        is_naflex=is_naflex,
        representation_mode=representation_mode,
        hierarchical_chunk_size=hierarchical_chunk_size,
        hierarchical_tile_batch_size=hierarchical_tile_batch_size,
        hierarchical_overview_max_num_patches=hierarchical_overview_max_num_patches,
        max_text_length=max_text_length,
    )
    cached = cache_features(frozen_features)
    model.train()
    with _device_autocast(device, dtype):
        output = model.forward_from_features(
            cached.frame_tokens,
            cached.frame_embeddings,
            cached.text_tokens,
            cached.text_embeddings,
            cached.text_mask,
            patch_valid_mask=cached.patch_valid_mask,
            spatial_shapes=cached.spatial_shapes,
            native_image_size=cached.native_image_size,
            processed_patch_grid=cached.processed_patch_grid,
            transform_hash=cached.transform_hash,
            token_coordinates=cached.token_coordinates,
            force_region_reduction=cached.force_region_reduction,
        )
        loss, text_to_pair_loss, pair_to_text_loss = (
            pair_balanced_symmetric_multi_positive_listwise_loss(
                output.score_matrix.float(),
                positive_mask,
                ignored_mask,
                query_indices,
            )
        )
    loss.backward()
    cached_grads = {
        "frame_tokens": cached.frame_tokens.grad,
        "frame_embeddings": cached.frame_embeddings.grad,
        "text_tokens": cached.text_tokens.grad,
        "text_embeddings": cached.text_embeddings.grad,
    }
    active_endpoint_names = [name for name, value in cached_grads.items() if value is not None]
    if not active_endpoint_names:
        raise RuntimeError(
            "FEATURE_ENDPOINT_NO_GRADIENT:all_feature_endpoints"
        )
    # The frozen model may intentionally select only the global PAIR path
    # (``final_v1_primary``), in which case text_tokens and/or auxiliary
    # visual endpoints are not part of the active objective.  Preserve the
    # exact active-path gradients and replay zero for inactive endpoints;
    # requiring every endpoint to receive a gradient would reject the
    # configured one-vector retrieval objective.
    replay_gradients = {
        name: (
            value.detach().clone()
            if value is not None
            else torch.zeros_like(getattr(cached, name))
        )
        for name, value in cached_grads.items()
    }
    feature_grads = CachedLogicalFeatures(
        frame_tokens=replay_gradients["frame_tokens"],
        frame_embeddings=replay_gradients["frame_embeddings"],
        text_tokens=replay_gradients["text_tokens"],
        text_embeddings=replay_gradients["text_embeddings"],
        text_mask=cached.text_mask,
        patch_valid_mask=cached.patch_valid_mask,
        spatial_shapes=cached.spatial_shapes,
        native_image_size=cached.native_image_size,
        processed_patch_grid=cached.processed_patch_grid,
        transform_hash=cached.transform_hash,
        token_coordinates=cached.token_coordinates,
        processing_mode=cached.processing_mode,
        force_region_reduction=cached.force_region_reduction,
    )
    if recompute_backbone:
        for start in range(0, len(pair_rows), physical_batch_size):
            end = start + physical_batch_size
            query_start = int(query_offsets[start])
            query_end = int(query_offsets[end])
            model.eval()
            features = encode_real_features(
                backbone,
                processor,
                pair_rows[start:end],
                query_rows[query_start:query_end],
                device,
                dtype=dtype,
                no_grad=False,
                max_num_patches=max_num_patches,
                is_naflex=is_naflex,
                representation_mode=representation_mode,
                hierarchical_chunk_size=hierarchical_chunk_size,
                hierarchical_tile_batch_size=hierarchical_tile_batch_size,
                hierarchical_overview_max_num_patches=hierarchical_overview_max_num_patches,
                max_text_length=max_text_length,
            )
            _feature_surrogate(
                features,
                feature_grads,
                start,
                end,
                query_start,
                query_end,
            ).backward()
        model.train()
    gradient_report = module_gradient_report(model)
    flat_gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not flat_gradients:
        raise RuntimeError("TRAINABLE_MODULE_NO_GRADIENT")
    flat = torch.cat(flat_gradients)
    if not torch.isfinite(flat).all():
        raise FloatingPointError("NONFINITE_GRADIENT")
    gradient_norm = float(flat.norm())
    scores = output.score_matrix.detach().float()
    positive_scores = scores.masked_select(positive_mask)
    negative_mask = ~positive_mask & ~ignored_mask
    negative_scores = scores.masked_select(negative_mask)
    weights = output.evidence.evidence_weights.detach().float()
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
    effective_tokens = entropy.exp()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "loss": float(loss.detach().cpu()),
        "text_to_pair_loss": float(text_to_pair_loss.detach().cpu()),
        "pair_to_text_loss": float(pair_to_text_loss.detach().cpu()),
        "score_shape": list(output.score_matrix.shape),
        "gradient_norm_preclip": gradient_norm,
        "recomputed_backbone": recompute_backbone,
        "representation_mode": representation_mode,
        "active_feature_gradient_endpoints": active_endpoint_names,
        "physical_microbatches": len(pair_rows) // physical_batch_size,
        "query_count": len(query_rows),
        "multi_positive_queries": int((positive_mask.sum(dim=1) > 1).sum()),
        "gradient_report": gradient_report,
        "embedding_diagnostics": {
            "pair_cls_norm_mean": float(output.pair_cls.detach().float().norm(dim=-1).mean()),
            "text_embedding_norm_mean": float(
                output.text_embedding.detach().float().norm(dim=-1).mean()
            ),
            "positive_score_mean": float(positive_scores.mean())
            if positive_scores.numel()
            else None,
            "negative_score_mean": float(negative_scores.mean())
            if negative_scores.numel()
            else None,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(unbiased=False)),
        },
        "evidence_diagnostics": {
            "entropy_mean": float(entropy.mean()),
            "effective_token_count_mean": float(effective_tokens.mean()),
            "near_uniform_fraction": float(
                (effective_tokens > (weights.shape[-1] * 0.9)).float().mean()
            ),
            "single_token_collapse_fraction": float(
                (effective_tokens < 2.0).float().mean()
            ),
            "evidence_gate": float(output.evidence.evidence_gate.detach()),
            "evidence_temperature": float(model.config.evidence_temperature),
        },
    }
