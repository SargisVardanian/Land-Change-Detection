from __future__ import annotations

import torch
from torch import Tensor, nn


class UnifiedRelevanceModel(nn.Module):
    """One scalar score with near-zero-anchored interaction pathways."""

    def __init__(self) -> None:
        super().__init__()
        # The global cosine and the causal evidence term are present at step
        # zero.  Change-slot interaction is a new pathway and is explicitly
        # anchored near zero so the initial model remains effectively the
        # accepted global retriever plus its existing evidence residual.
        self.slot_scale = nn.Parameter(torch.tensor(1e-4))

    def forward(
        self,
        global_score: Tensor,
        evidence_score: Tensor,
        slot_score: Tensor,
    ) -> Tensor:
        if (
            global_score.shape != evidence_score.shape
            or global_score.shape != slot_score.shape
        ):
            raise ValueError("all relevance pathways must share [queries, pairs] shape")
        return global_score + evidence_score + self.slot_scale * slot_score
