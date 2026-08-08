"""Resolution-flexible and synchronized temporal-view contracts."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

import torch
from PIL import Image
from torch import Tensor


def synchronized_transform_hash(
    *,
    native_size: tuple[int, int],
    processed_grid: tuple[int, int],
    max_num_patches: int,
    patch_size: int,
) -> str:
    """Hash geometry shared by T1 and T2 preprocessing."""

    payload = "|".join(
        str(value)
        for value in (
            native_size,
            processed_grid,
            max_num_patches,
            patch_size,
        )
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def assert_synchronized_pair_views(
    t1: Image.Image,
    t2: Image.Image,
    *,
    transform_hash_t1: str | None = None,
    transform_hash_t2: str | None = None,
) -> None:
    """Reject independently transformed temporal views."""

    if t1.size != t2.size:
        raise ValueError("T1/T2 native image sizes differ")
    if t1.mode != t2.mode:
        raise ValueError("T1/T2 image modes differ")
    if (transform_hash_t1 is None) != (transform_hash_t2 is None):
        raise ValueError("one temporal view is missing its transform hash")
    if transform_hash_t1 is not None and transform_hash_t1 != transform_hash_t2:
        raise ValueError("T1/T2 transform hashes differ")


def validate_patch_budget_sequence(
    budgets: Iterable[int], *, supported: Iterable[int]
) -> str:
    """Return a deterministic budget-sequence hash after strict validation."""

    allowed = {int(value) for value in supported}
    values = [int(value) for value in budgets]
    if not values or any(value not in allowed for value in values):
        raise ValueError("every patch budget must be explicitly supported")
    return sha256(",".join(map(str, values)).encode("utf-8")).hexdigest()


def patch_mask_from_spatial_shapes(
    spatial_shapes: Tensor, *, token_count: int
) -> Tensor:
    """Build packed-patch validity masks from SigLIP2 patch-grid shapes."""

    if spatial_shapes.ndim not in (2, 3) or spatial_shapes.shape[-1] != 2:
        raise ValueError("spatial_shapes must end in a height/width pair")
    shapes = spatial_shapes.to(dtype=torch.long)
    counts = shapes[..., 0] * shapes[..., 1]
    if torch.any(counts <= 0) or torch.any(counts > token_count):
        raise ValueError("spatial shape does not fit token_count")
    index = torch.arange(token_count, device=spatial_shapes.device)
    return index.view(*([1] * (counts.ndim)), token_count) < counts.unsqueeze(-1)
