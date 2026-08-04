"""One unified scalar relevance model."""

from __future__ import annotations

import torch

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class RelevanceOutput:
    score: Tensor
    global_logit: Tensor
    evidence_logit: Tensor
    slot_logit: Tensor
    late_feature: Tensor


class UnifiedRelevanceModel(nn.Module):
    """Global score plus near-zero residual evidence pathways.

    The gates are intentionally 1e-3 rather than exactly zero: this keeps the
    step-zero score numerically equivalent to the global baseline while allowing
    the first backward pass to reach the evidence pathway.
    """

    def __init__(self, hidden_dim: int = 512, gate_init: float = 1e-3):
        super().__init__()
        self.evidence_gate = nn.Parameter(Tensor([gate_init]))
        self.slot_gate = nn.Parameter(Tensor([gate_init]))
        self.late_gate = nn.Parameter(Tensor([gate_init]))
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        global_logit: Tensor,
        evidence_logit: Tensor,
        slot_logit: Tensor,
        late_feature: Tensor,
        text_cls: Tensor,
        sequence_cls: Tensor,
        evidence_vector: Tensor,
    ) -> RelevanceOutput:
        if global_logit.shape != evidence_logit.shape or global_logit.shape != slot_logit.shape:
            raise ValueError("relevance components must share [Q,P] shape")
        residual_input = torch.cat(
            [
                text_cls[:, None, :].expand(-1, sequence_cls.shape[0], -1),
                sequence_cls[None, :, :].expand(text_cls.shape[0], -1, -1),
                evidence_vector,
            ],
            dim=-1,
        )
        residual = self.residual(residual_input).squeeze(-1)
        score = global_logit + self.evidence_gate * evidence_logit + self.slot_gate * slot_logit + self.late_gate * late_feature + residual
        return RelevanceOutput(score, global_logit, evidence_logit, slot_logit, late_feature)
