from __future__ import annotations

import math
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


def normalize_patch_metadata(
    frame_tokens: Tensor,
    patch_valid_mask: Tensor | None = None,
    spatial_shapes: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Validate and normalize variable-length NaFlex patch metadata.

    NaFlex pads every image to the configured patch budget.  The returned mask
    is therefore the only authoritative indicator of image content; callers
    must not read padded token values.  ``spatial_shapes`` is in patch-grid
    coordinates (height, width), as returned by the SigLIP2 processor.
    """

    require_rank(frame_tokens, 4, "frame_tokens")
    b, t, n, _ = frame_tokens.shape
    if patch_valid_mask is not None:
        require_rank(patch_valid_mask, 3, "patch_valid_mask")
        if patch_valid_mask.shape != (b, t, n):
            raise ValueError("patch_valid_mask must be [B,T,N] and align with tokens")
        patch_valid_mask = patch_valid_mask.to(device=frame_tokens.device, dtype=torch.bool)
    if spatial_shapes is not None:
        require_rank(spatial_shapes, 3, "spatial_shapes")
        if spatial_shapes.shape != (b, t, 2):
            raise ValueError("spatial_shapes must be [B,T,2]")
        spatial_shapes = spatial_shapes.to(device=frame_tokens.device, dtype=torch.long)
        if torch.any(spatial_shapes <= 0):
            raise ValueError("spatial_shapes must be positive patch-grid dimensions")
        if torch.any(spatial_shapes[..., 0] * spatial_shapes[..., 1] > n):
            raise ValueError("spatial_shapes exceed the padded patch-token budget")

    if patch_valid_mask is None and spatial_shapes is None:
        side = math.isqrt(n)
        grid = (side, side) if side * side == n else (1, n)
        spatial_shapes = torch.tensor(
            grid, dtype=torch.long, device=frame_tokens.device
        ).view(1, 1, 2).expand(b, t, 2).clone()
        patch_valid_mask = torch.ones(
            (b, t, n), dtype=torch.bool, device=frame_tokens.device
        )
    elif spatial_shapes is None:
        counts = patch_valid_mask.sum(dim=-1)
        shapes: list[tuple[int, int]] = []
        for count in counts.detach().cpu().reshape(-1).tolist():
            side = math.isqrt(int(count))
            shapes.append((side, side) if side * side == int(count) else (1, int(count)))
        spatial_shapes = torch.tensor(
            shapes, dtype=torch.long, device=frame_tokens.device
        ).reshape(b, t, 2)
    elif patch_valid_mask is None:
        counts = spatial_shapes[..., 0] * spatial_shapes[..., 1]
        patch_valid_mask = torch.arange(n, device=frame_tokens.device).view(1, 1, n) < counts.unsqueeze(-1)
    else:
        expected = spatial_shapes[..., 0] * spatial_shapes[..., 1]
        actual = patch_valid_mask.sum(dim=-1)
        if not torch.equal(expected, actual):
            raise ValueError("patch_valid_mask counts disagree with spatial_shapes")

    if torch.any(patch_valid_mask.sum(dim=-1) == 0):
        raise ValueError("every frame must contain at least one valid patch token")
    return patch_valid_mask, spatial_shapes


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
