from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def require_rank(value: Tensor, rank: int, name: str) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tuple(value.shape)}")


def require_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def validate_feature_contract(
    frame_tokens: Tensor,
    frame_embeddings: Tensor,
    text_tokens: Tensor,
    text_embeddings: Tensor,
    text_mask: Tensor,
    *,
    hidden_size: int,
    expected_patch_tokens: int | None = None,
) -> None:
    for value, rank, name in (
        (frame_tokens, 4, "frame_tokens"),
        (frame_embeddings, 3, "frame_embeddings"),
        (text_tokens, 3, "text_tokens"),
        (text_embeddings, 2, "text_embeddings"),
        (text_mask, 2, "text_mask"),
    ):
        require_rank(value, rank, name)
    b, t, n, d = frame_tokens.shape
    if frame_embeddings.shape != (b, t, d):
        raise ValueError("frame_embeddings must align with frame_tokens")
    if (
        text_tokens.shape[0] != text_embeddings.shape[0]
        or text_tokens.shape[:2] != text_mask.shape
    ):
        raise ValueError("text token, pooled embedding and mask shapes disagree")
    if (
        d != hidden_size
        or text_tokens.shape[-1] != hidden_size
        or text_embeddings.shape[-1] != hidden_size
    ):
        raise ValueError("all SigLIP-2 feature dimensions must equal hidden_size")
    if expected_patch_tokens is not None and n != expected_patch_tokens:
        raise ValueError(
            f"native patch count changed: expected {expected_patch_tokens}, got {n}"
        )
    if t < 2:
        raise ValueError("temporal retrieval requires at least two frames")


def validate_relevance_masks(
    positive_mask: Tensor,
    ignored_mask: Tensor,
    *,
    score_shape: tuple[int, int] | None = None,
) -> None:
    if positive_mask.dtype != torch.bool or ignored_mask.dtype != torch.bool:
        raise TypeError("relevance masks must be bool tensors")
    if positive_mask.shape != ignored_mask.shape:
        raise ValueError("positive and ignored masks must have equal shape")
    if score_shape is not None and tuple(positive_mask.shape) != score_shape:
        raise ValueError("relevance masks do not match score matrix")
    if torch.any(positive_mask & ignored_mask):
        raise ValueError("a pair cannot be positive and ignored")
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("every query needs at least one positive pair")
    if torch.any((~ignored_mask).sum(dim=1) == 0):
        raise ValueError("every query needs at least one valid candidate")


def tensor_shape(value: Tensor) -> list[int]:
    return [int(x) for x in value.shape]


def module_parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "parameter_count": sum(p.numel() for p in module.parameters()),
        "trainable_count": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return tensor_shape(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
