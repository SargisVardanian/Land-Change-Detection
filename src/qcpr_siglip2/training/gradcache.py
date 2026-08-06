"""Exact logical-batch feature caching for temporal SigLIP-2 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from ..data.runtime import RawFeatureBatch, _device_autocast, encode_real_features
from ..models.model import Siglip2TemporalRetrievalModel
from .objective import multi_positive_listwise_loss


@dataclass(frozen=True)
class CachedLogicalFeatures:
    frame_tokens: Tensor
    frame_embeddings: Tensor
    text_tokens: Tensor
    text_embeddings: Tensor
    text_mask: Tensor


def module_gradient_report(
    model: Siglip2TemporalRetrievalModel,
) -> dict[str, dict[str, int | float | bool]]:
    """Audit trainable and frozen gradient state by model component."""

    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "temporal_adapter": [],
        "evidence_bottleneck": [],
        "retrieval_temperature": [],
        "siglip2_vision_backbone": [],
        "siglip2_text_backbone": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("temporal_adapter."):
            bucket = "temporal_adapter"
        elif name.startswith("evidence_bottleneck."):
            bucket = "evidence_bottleneck"
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
    )


def _feature_surrogate(
    features: RawFeatureBatch,
    feature_grads: CachedLogicalFeatures,
    pair_start: int,
    pair_end: int,
    captions_per_pair: int,
) -> Tensor:
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
    captions_per_pair: int,
    dtype: torch.dtype,
    no_grad: bool,
) -> RawFeatureBatch:
    """Encode one logical batch through bounded physical microbatches.

    The listwise objective is still evaluated once after concatenation.  The
    chunking only bounds frozen-backbone activation and decoder input memory;
    it must never turn the logical batch into independent losses.
    """

    if len(pair_rows) % physical_batch_size:
        raise ValueError("logical pair count must divide physical microbatch size")
    chunks: list[RawFeatureBatch] = []
    for start in range(0, len(pair_rows), physical_batch_size):
        end = start + physical_batch_size
        query_start = start * captions_per_pair
        query_end = end * captions_per_pair
        chunks.append(
            encode_real_features(
                backbone,
                processor,
                pair_rows[start:end],
                query_rows[query_start:query_end],
                device,
                dtype=dtype,
                no_grad=no_grad,
            )
        )
    if not chunks:
        raise ValueError("logical batch must contain at least one pair")
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
    captions_per_pair: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    dtype: torch.dtype = torch.bfloat16,
    gradient_clip_norm: float = 1.0,
    recompute_backbone: bool = False,
) -> dict[str, Any]:
    """Run one common logical score matrix and exact feature-gradient replay.

    The listwise loss is evaluated once over the complete logical batch.  When
    the top pretrained blocks are trainable, feature gradients are replayed
    through physical microbatches (GradCache-style); no independent
    microbatch loss is substituted for the logical objective.
    """

    if len(pair_rows) % physical_batch_size:
        raise ValueError("logical pair count must divide physical microbatch size")
    if len(query_rows) != len(pair_rows) * captions_per_pair:
        raise ValueError("query rows do not match pair/caption contract")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    frozen_features = _encode_logical_features_in_chunks(
        backbone,
        processor,
        pair_rows,
        query_rows,
        device=device,
        physical_batch_size=physical_batch_size,
        captions_per_pair=captions_per_pair,
        dtype=dtype,
        no_grad=True,
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
        )
        loss = multi_positive_listwise_loss(
            output.score_matrix.float(), positive_mask, ignored_mask
        )
    loss.backward()
    cached_grads = {
        "frame_tokens": cached.frame_tokens.grad,
        "frame_embeddings": cached.frame_embeddings.grad,
        "text_tokens": cached.text_tokens.grad,
        "text_embeddings": cached.text_embeddings.grad,
    }
    missing = [name for name, value in cached_grads.items() if value is None]
    if missing:
        raise RuntimeError(
            "FEATURE_ENDPOINT_NO_GRADIENT:" + ",".join(sorted(missing))
        )
    feature_grads = CachedLogicalFeatures(
        frame_tokens=cast(Tensor, cached_grads["frame_tokens"]).detach().clone(),
        frame_embeddings=cast(
            Tensor, cached_grads["frame_embeddings"]
        ).detach().clone(),
        text_tokens=cast(Tensor, cached_grads["text_tokens"]).detach().clone(),
        text_embeddings=cast(
            Tensor, cached_grads["text_embeddings"]
        ).detach().clone(),
        text_mask=cached.text_mask,
    )
    if recompute_backbone:
        for start in range(0, len(pair_rows), physical_batch_size):
            end = start + physical_batch_size
            model.eval()
            features = encode_real_features(
                backbone,
                processor,
                pair_rows[start:end],
                query_rows[start * captions_per_pair : end * captions_per_pair],
                device,
                dtype=dtype,
                no_grad=False,
            )
            _feature_surrogate(
                features,
                feature_grads,
                start,
                end,
                captions_per_pair,
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
        "score_shape": list(output.score_matrix.shape),
        "gradient_norm_preclip": gradient_norm,
        "recomputed_backbone": recompute_backbone,
        "physical_microbatches": len(pair_rows) // physical_batch_size,
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
