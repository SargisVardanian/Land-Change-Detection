"""The single primary listwise retrieval objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossOutput:
    loss: Tensor
    positive_count: int
    valid_query_count: int
    mean_positive_score: Tensor
    mean_all_score: Tensor

    @property
    def primary_scalar_count(self) -> int:
        return 1


class UnifiedListwiseLoss(nn.Module):
    """Multi-positive graded listwise loss with exactly one backprop scalar."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        grade_weights: tuple[float, float, float, float] = (0.0, 1.0, 1.25, 1.5),
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if len(grade_weights) != 4 or grade_weights[0] != 0.0:
            raise ValueError("grade_weights must define grades 0..3 with grade 0 weight 0")
        if any(weight < 0.0 for weight in grade_weights[1:]):
            raise ValueError("positive grade weights must be non-negative")
        self.temperature = float(temperature)
        self.register_buffer("grade_weights", torch.tensor(grade_weights, dtype=torch.float32), persistent=False)

    def forward(self, scores: Tensor, grades: Tensor, valid_mask: Tensor | None = None) -> LossOutput:
        if scores.ndim != 2 or grades.shape != scores.shape:
            raise ValueError("scores and grades must share [Q,P] shape")
        if valid_mask is None:
            valid_mask = torch.ones_like(grades, dtype=torch.bool)
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must match scores")
        positive = (grades > 0) & valid_mask
        query_has_positive = positive.any(dim=1)
        if not bool(query_has_positive.all()):
            missing = int((~query_has_positive).sum())
            raise ValueError(f"{missing} queries have no valid positive")
        scaled = scores / self.temperature
        denominator_logits = scaled.masked_fill(~valid_mask, torch.finfo(scaled.dtype).min)
        weights = self.grade_weights.to(device=grades.device)[grades.clamp(0, 3)]
        numerator_logits = (scaled + torch.log(weights.clamp_min(torch.finfo(scaled.dtype).tiny))).masked_fill(~positive, torch.finfo(scaled.dtype).min)
        loss_per_query = torch.logsumexp(denominator_logits, dim=1) - torch.logsumexp(numerator_logits, dim=1)
        loss = loss_per_query[query_has_positive].mean()
        positive_scores = scores[positive]
        return LossOutput(
            loss=loss,
            positive_count=int(positive.sum()),
            valid_query_count=int(query_has_positive.sum()),
            mean_positive_score=positive_scores.mean(),
            mean_all_score=scores[valid_mask].mean(),
        )
