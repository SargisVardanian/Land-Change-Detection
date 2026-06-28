from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import torch
from torch import Tensor
from torch.nn import functional as F

from land_change_detection.losses.unichange_losses import dice_loss


@dataclass(frozen=True)
class EventMatchResult:
    event_indices: Tensor
    component_indices: Tensor
    cost: Tensor


def _bce_per_pair(mask_logits: Tensor, components: Tensor) -> Tensor:
    logits = mask_logits[:, None, :].expand(-1, components.shape[0], -1)
    targets = components[None, :, :].expand(mask_logits.shape[0], -1, -1).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)


def _dice_per_pair(mask_logits: Tensor, components: Tensor) -> Tensor:
    costs = torch.zeros(mask_logits.shape[0], components.shape[0], device=mask_logits.device, dtype=mask_logits.dtype)
    for event_index in range(mask_logits.shape[0]):
        for component_index in range(components.shape[0]):
            costs[event_index, component_index] = dice_loss(
                mask_logits[event_index : event_index + 1],
                components[component_index : component_index + 1],
            )
    return costs


def event_component_cost(mask_logits: Tensor, components: Tensor, dice_weight: float = 1.0, bce_weight: float = 1.0) -> Tensor:
    if mask_logits.ndim != 2 or components.ndim != 2:
        raise ValueError("mask_logits and components must have shape [K/R,N].")
    if components.shape[0] == 0:
        return torch.zeros(mask_logits.shape[0], 0, device=mask_logits.device, dtype=mask_logits.dtype)
    if mask_logits.shape[1] != components.shape[1]:
        raise ValueError("mask_logits and components must share flattened mask size.")
    return dice_weight * _dice_per_pair(mask_logits, components) + bce_weight * _bce_per_pair(mask_logits, components)


def _bruteforce_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    num_events, num_components = cost.shape
    if num_components == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    if num_components > num_events:
        raise ValueError("Cannot match more components than event queries.")
    best_perm: tuple[int, ...] | None = None
    best_cost: Tensor | None = None
    for perm in permutations(range(num_events), num_components):
        candidate = cost[torch.tensor(perm, device=cost.device), torch.arange(num_components, device=cost.device)].sum()
        if best_cost is None or candidate.item() < best_cost.item():
            best_cost = candidate
            best_perm = perm
    assert best_perm is not None
    return torch.tensor(best_perm, dtype=torch.long, device=cost.device), torch.arange(num_components, dtype=torch.long, device=cost.device)


def hungarian_match_events(mask_logits: Tensor, components: Tensor, dice_weight: float = 1.0, bce_weight: float = 1.0) -> EventMatchResult:
    cost = event_component_cost(mask_logits, components.to(mask_logits.device), dice_weight=dice_weight, bce_weight=bce_weight)
    if cost.shape[1] == 0:
        empty = torch.empty(0, dtype=torch.long, device=mask_logits.device)
        return EventMatchResult(empty, empty, cost)
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        event_indices = torch.tensor(row, dtype=torch.long, device=mask_logits.device)
        component_indices = torch.tensor(col, dtype=torch.long, device=mask_logits.device)
    except Exception:
        event_indices, component_indices = _bruteforce_assignment(cost)
    return EventMatchResult(event_indices=event_indices, component_indices=component_indices, cost=cost)
