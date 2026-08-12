"""Resolution-flexible and synchronized temporal-view contracts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256

import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class PatchBudgetAssignment:
    exposure_index: int
    cycle_index: int
    item_id: str
    source: str
    max_num_patches: int


@dataclass(frozen=True)
class PatchBudgetSchedule:
    assignments: tuple[PatchBudgetAssignment, ...]
    pair_sequence_sha256: str
    budget_sequence_sha256: str
    schedule_sha256: str


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


def build_patch_budget_schedule(
    item_ids: Iterable[str],
    sources: Iterable[str],
    *,
    budgets: Iterable[int] = (256, 576, 1024),
    cycles: int = 3,
    seed: int = 0,
) -> PatchBudgetSchedule:
    """Assign every physical item every budget without changing item order.

    The pair sequence is repeated exactly ``cycles`` times. A stable per-item
    offset rotates budget order, while a full set of cycles guarantees that
    patch count cannot be a deterministic proxy for source identity.
    """

    ids = [str(value) for value in item_ids]
    source_values = [str(value) for value in sources]
    budget_values = tuple(int(value) for value in budgets)
    if not ids or len(ids) != len(source_values):
        raise ValueError("item_ids and sources must be non-empty and aligned")
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids must be unique within the base schedule")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    validate_patch_budget_sequence(budget_values, supported=budget_values)
    assignments: list[PatchBudgetAssignment] = []
    for cycle_index in range(cycles):
        for item_id, source in zip(ids, source_values):
            offset_digest = sha256(f"{seed}|{item_id}".encode()).digest()
            offset = int.from_bytes(offset_digest[:8], "big") % len(budget_values)
            assignments.append(
                PatchBudgetAssignment(
                    exposure_index=len(assignments),
                    cycle_index=cycle_index,
                    item_id=item_id,
                    source=source,
                    max_num_patches=budget_values[
                        (offset + cycle_index) % len(budget_values)
                    ],
                )
            )
    pair_sequence = [assignment.item_id for assignment in assignments]
    budget_sequence = [assignment.max_num_patches for assignment in assignments]
    payload = [asdict(assignment) for assignment in assignments]
    return PatchBudgetSchedule(
        assignments=tuple(assignments),
        pair_sequence_sha256=sha256(
            "\n".join(pair_sequence).encode("utf-8")
        ).hexdigest(),
        budget_sequence_sha256=sha256(
            "\n".join(map(str, budget_sequence)).encode("utf-8")
        ).hexdigest(),
        schedule_sha256=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    )


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
