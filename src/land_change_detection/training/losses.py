from __future__ import annotations

import math
from typing import Any


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def info_nce_loss(anchor: list[float], positive: list[float], negatives: list[list[float]], temperature: float = 0.1) -> float:
    pos_score = math.exp(_dot(anchor, positive) / temperature)
    neg_score = sum(math.exp(_dot(anchor, negative) / temperature) for negative in negatives)
    denom = pos_score + neg_score
    if denom <= 0.0:
        return 0.0
    return -math.log(pos_score / denom)


def triplet_ranking_loss(anchor: list[float], positive: list[float], negative: list[float], margin: float = 0.2) -> float:
    pos_score = _dot(anchor, positive)
    neg_score = _dot(anchor, negative)
    return max(0.0, margin - pos_score + neg_score)


def semantic_change_loss(
    output: Any,
    before_target: Any,
    after_target: Any,
    change_target: Any | None = None,
    *,
    semantic_weight: float = 1.0,
    change_weight: float = 0.3,
    ignore_index: int = -1,
) -> Any:
    """Composite loss for semantic-first land-cover change training.

    `output` is intentionally duck-typed to keep this module importable in
    lightweight retrieval-only contexts; it must expose before/after semantic
    logits and one-channel change logits.
    """

    import torch
    from torch.nn import functional as F

    before_target = before_target.long()
    after_target = after_target.long()
    valid_semantic = (before_target != ignore_index) & (after_target != ignore_index)
    if torch.any(valid_semantic):
        before_loss = F.cross_entropy(output.before_logits, before_target, ignore_index=ignore_index)
        after_loss = F.cross_entropy(output.after_logits, after_target, ignore_index=ignore_index)
        semantic_loss = (before_loss + after_loss) * 0.5
    else:
        semantic_loss = output.before_logits.sum() * 0.0

    if change_target is None:
        valid = valid_semantic
        change_target = ((before_target != after_target) & valid).to(dtype=output.change_logits.dtype)
        valid_change = valid.unsqueeze(1)
    else:
        change_target = change_target.to(dtype=output.change_logits.dtype)
        valid_change = change_target >= 0

    if change_target.ndim == 3:
        change_target = change_target.unsqueeze(1)
    if valid_change.ndim == 3:
        valid_change = valid_change.unsqueeze(1)
    if torch.any(valid_change):
        change_loss = F.binary_cross_entropy_with_logits(
            output.change_logits[valid_change],
            change_target[valid_change],
        )
    else:
        change_loss = output.change_logits.sum() * 0.0

    return semantic_weight * semantic_loss + change_weight * change_loss
