from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ContributionSelection:
    mask: Tensor
    status: str
    selected_count: int
    positive_mass: float


def directional_counterfactuals(
    images: Tensor, *, grid: int, direction: str, replacement: str = "raw"
) -> tuple[Tensor, list[tuple[int, int, int, int]]]:
    """Remove an appeared/disappeared change one spatial tile at a time."""
    if images.ndim != 4 or images.shape[0] != 2:
        raise ValueError("images must have shape [2,C,H,W]")
    if grid <= 0:
        raise ValueError("grid must be positive")
    if direction not in {"appeared", "disappeared"}:
        raise ValueError("direction must be appeared or disappeared")
    _, _, height, width = images.shape
    variants, boxes = [], []
    for row in range(grid):
        y0, y1 = row * height // grid, (row + 1) * height // grid
        for column in range(grid):
            x0, x1 = column * width // grid, (column + 1) * width // grid
            candidate = images.clone()
            _replace_tile(candidate, images, y0, y1, x0, x1, direction=direction, replacement=replacement)
            variants.append(candidate)
            boxes.append((y0, y1, x0, x1))
    return torch.stack(variants), boxes


def apply_selected_counterfactual(
    images: Tensor, boxes: list[tuple[int, int, int, int]], selected: Tensor, *, direction: str, replacement: str = "raw"
) -> Tensor:
    if selected.numel() != len(boxes):
        raise ValueError("selected mask and boxes must have the same tile count")
    result = images.clone()
    for active, (y0, y1, x0, x1) in zip(selected.flatten().tolist(), boxes, strict=True):
        if not active:
            continue
        _replace_tile(result, images, y0, y1, x0, x1, direction=direction, replacement=replacement)
    return result


def _replace_tile(
    result: Tensor, images: Tensor, y0: int, y1: int, x0: int, x1: int, *, direction: str, replacement: str
) -> None:
    if direction not in {"appeared", "disappeared"}:
        raise ValueError("direction must be appeared or disappeared")
    if replacement not in {"raw", "color_matched"}:
        raise ValueError("replacement must be raw or color_matched")
    destination, source = (1, 0) if direction == "appeared" else (0, 1)
    source_tile = images[source, :, y0:y1, x0:x1]
    target_tile = images[destination, :, y0:y1, x0:x1]
    if replacement == "color_matched":
        source_mean = source_tile.mean(dim=(-2, -1), keepdim=True)
        source_std = source_tile.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        target_mean = target_tile.mean(dim=(-2, -1), keepdim=True)
        target_std = target_tile.std(dim=(-2, -1), keepdim=True)
        source_tile = (source_tile - source_mean) / source_std * target_std + target_mean
    result[destination, :, y0:y1, x0:x1] = source_tile


def contribution_mass_selection(evidence: Tensor, *, mass: float = 0.8) -> ContributionSelection:
    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1]")
    values = evidence.clamp_min(0).flatten()
    total = values.sum()
    if not torch.isfinite(total):
        raise ValueError("evidence must be finite")
    if float(total) <= 1e-8:
        return ContributionSelection(
            torch.zeros_like(evidence, dtype=torch.bool),
            "NO_FAITHFUL_SPATIAL_EVIDENCE", 0, 0.0,
        )
    order = values.argsort(descending=True)
    cumulative = values[order].cumsum(0)
    count = int(torch.searchsorted(cumulative, mass * total).item()) + 1
    selected = torch.zeros_like(values, dtype=torch.bool)
    selected[order[:count]] = True
    return ContributionSelection(selected.reshape_as(evidence), "POSITIVE_SPATIAL_EVIDENCE", count, float(total))


def reconstruct_final_score(scores, rerank_logits: Tensor) -> Tensor:
    weights = F.softplus(rerank_logits)
    reconstructed = scores.global_score + weights[0] * scores.local_score + weights[1] * scores.token_patch_score
    torch.testing.assert_close(reconstructed, scores.reranked_score, atol=1e-5, rtol=1e-5)
    return reconstructed


@torch.inference_mode()
def score_variants(model, images: Tensor, query: str, temporal_valid_mask: Tensor, *, chunk_size: int = 8) -> Tensor:
    if images.ndim != 5 or images.shape[1] != 2:
        raise ValueError("images must have shape [B,2,C,H,W]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = next(model.parameters()).device
    scores = []
    for chunk in images.split(chunk_size):
        chunk = chunk.to(device)
        valid = temporal_valid_mask.to(device).reshape(1, -1).expand(chunk.shape[0], -1)
        output = model(chunk, [query], temporal_valid_mask=valid)
        reconstruct_final_score(output.scores, model.grounder.rerank_logits)
        scores.append(output.scores.reranked_score[0].float().cpu())
    return torch.cat(scores)


@torch.inference_mode()
def counterfactual_evidence(
    score_fn: Callable[[Tensor], Tensor], images: Tensor, *, grid: int, direction: str, chunk_size: int = 8, replacement: str = "raw"
) -> tuple[Tensor, Tensor, list[tuple[int, int, int, int]]]:
    original = score_fn(images.unsqueeze(0)).reshape(-1)[0]
    variants, boxes = directional_counterfactuals(images, grid=grid, direction=direction, replacement=replacement)
    modified = torch.cat([score_fn(chunk).reshape(-1) for chunk in variants.split(chunk_size)])
    signed = (original - modified).reshape(grid, grid)
    return original, signed, boxes
