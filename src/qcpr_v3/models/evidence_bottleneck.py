"""Causal query-conditioned sparse evidence bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sparsemax(values: Tensor, dim: int = -1) -> Tensor:
    values = values - values.amax(dim=dim, keepdim=True)
    sorted_values, _ = torch.sort(values, dim=dim, descending=True)
    cumulative = sorted_values.cumsum(dim)
    positions = torch.arange(1, values.shape[dim] + 1, device=values.device, dtype=values.dtype)
    shape = [1] * values.ndim
    shape[dim] = values.shape[dim]
    support = 1.0 + positions.reshape(shape) * sorted_values > cumulative
    support_count = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_index = (support_count - 1).to(torch.long)
    tau = (cumulative.gather(dim, threshold_index) - 1.0) / support_count.to(values.dtype)
    return torch.clamp(values - tau, min=0.0)


@dataclass(frozen=True)
class EvidenceOutput:
    evidence_vector: Tensor
    relevance_logits: Tensor
    relevance_weights: Tensor
    temporal_shape: tuple[int, int]
    token_similarity: Tensor | None = None


@dataclass(frozen=True)
class PairwiseEvidenceOutput:
    evidence_vector: Tensor
    relevance_logits: Tensor
    relevance_weights: Tensor
    late_score: Tensor


class EvidenceBottleneck(nn.Module):
    def __init__(self, hidden_dim: int = 512, *, normalizer: str = "sparsemax", temperature: float = 0.07):
        super().__init__()
        if normalizer not in {"sparsemax", "softmax"}:
            raise ValueError("normalizer must be sparsemax or softmax")
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.visual_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalizer = normalizer
        self.temperature = float(temperature)
        for layer in (self.query_projection, self.visual_projection, self.output_projection):
            nn.init.eye_(layer.weight)

    def _safe_mask(self, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            mask = mask.clone()
            empty = ~mask.any(dim=-1)
            mask[empty, 0] = True
        return mask

    def _normalize(self, logits: Tensor, mask: Tensor) -> Tensor:
        safe_mask = self._safe_mask(mask)
        masked = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        if self.normalizer == "softmax":
            weights = F.softmax(masked, dim=-1)
        else:
            weights = sparsemax(masked, dim=-1)
        return weights * safe_mask.to(weights.dtype)

    def forward(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        dense_tokens: Tensor,
        token_mask: Tensor,
    ) -> EvidenceOutput:
        if text_tokens.ndim != 3 or dense_tokens.ndim != 4:
            raise ValueError("text_tokens [B,L,D] and dense_tokens [B,T,N,D] are required")
        batch, time_count, n, dim = dense_tokens.shape
        if text_mask.shape != text_tokens.shape[:2] or token_mask.shape != (batch, time_count, n):
            raise ValueError("evidence masks have invalid shapes")
        query = F.normalize(self.query_projection(text_tokens), dim=-1)
        visual = F.normalize(self.visual_projection(dense_tokens), dim=-1)
        similarity = torch.einsum("bld,btnd->bltn", query, visual) / self.temperature
        text_weights = text_mask.to(similarity.dtype)[:, :, None, None]
        logits = (similarity * text_weights).sum(dim=1) / text_weights.sum(dim=1).clamp_min(1.0)
        weights = self._normalize(logits.reshape(batch, time_count * n), token_mask.reshape(batch, time_count * n))
        weights = weights.reshape(batch, time_count, n)
        vector = torch.einsum("btn,btnd->bd", weights, self.output_projection(dense_tokens))
        return EvidenceOutput(vector, logits, weights, (time_count, n), similarity)

    def pairwise(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        dense_tokens: Tensor,
        token_mask: Tensor,
    ) -> PairwiseEvidenceOutput:
        """Memory-bounded evidence for a Q by P logical batch.

        The token-level interaction is linearly aggregated over valid query
        tokens before the QxP token map is built. The exact token map is used
        by forward() for stage-2 Top-K reranking.
        """
        if text_tokens.ndim != 3 or dense_tokens.ndim != 4:
            raise ValueError("pairwise inputs have invalid rank")
        query = self.query_projection(text_tokens)
        query = (query * text_mask.to(query.dtype).unsqueeze(-1)).sum(dim=1)
        query = query / text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(query.dtype)
        query = F.normalize(query, dim=-1)
        visual = F.normalize(self.visual_projection(dense_tokens), dim=-1)
        logits = torch.einsum("qd,ptnd->qptn", query, visual) / self.temperature
        valid = token_mask.unsqueeze(0).expand(text_tokens.shape[0], -1, -1, -1)
        weights = self._normalize(logits.reshape(text_tokens.shape[0], dense_tokens.shape[0], -1), valid.reshape(text_tokens.shape[0], dense_tokens.shape[0], -1))
        weights = weights.reshape_as(logits)
        vector = torch.einsum("qptn,ptnd->qpd", weights, self.output_projection(dense_tokens))
        late_score = logits.amax(dim=-1).mean(dim=-1)
        return PairwiseEvidenceOutput(vector, logits, weights, late_score)
