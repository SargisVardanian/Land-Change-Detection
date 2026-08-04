"""Hierarchical Retrieval and Grounding (QCPR-HRG).

This module is a contract-level architecture for the next controlled ablation.
It keeps one query-independent global vector for ANN indexing, preserves
compressed temporal tokens for candidate reranking, and derives a
query-conditioned evidence map from the same token similarities.

It deliberately contains no mask loss and no full-gallery cross-attention.
UniverSat/Jina wrappers remain outside this module and may be frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .qcpr_single_pass import DeepResidualPairAdapter, SinglePassConfig


@dataclass(frozen=True)
class QCPRHRGConfig:
    """Stable defaults for the bounded architecture probe."""

    visual_dim: int = 768
    retrieval_dim: int = 512
    token_dim: int = 512
    token_grid: int = 8
    context_grid: int = 4
    slot_count: int = 8
    slot_heads: int = 8
    pair_layers: int = 3
    pair_heads: int = 12
    token_adapter_layers: int = 2
    token_bottleneck_ratio: int = 4
    ffn_ratio: int = 4
    dropout: float = 0.0
    evidence_topk: int = 4
    max_logit_scale: float = 100.0


@dataclass(frozen=True)
class HRGVisualEncoding:
    """Indexable global vector plus candidate-stage visual evidence."""

    global_pair_vector: Tensor
    change_slot_vectors: Tensor
    dense_change_tokens: Tensor
    pyramid_tokens: Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HRGRetrievalOutput:
    scores: Tensor
    global_scores: Tensor
    slot_scores: Tensor
    late_scores: Tensor
    evidence_scores: Tensor
    evidence_logits: Tensor
    evidence_probabilities: Tensor
    visual: HRGVisualEncoding

    @property
    def evidence_map(self) -> Tensor:
        grid = self.visual.metadata["evidence_grid"]
        return self.evidence_logits.reshape(
            self.evidence_logits.shape[0],
            self.evidence_logits.shape[1],
            grid[0],
            grid[1],
        )

    @property
    def global_pair_vectors(self) -> Tensor:
        return self.visual.global_pair_vector


class TemporalDescriptorBuilder(nn.Module):
    """Build signed temporal descriptors for T=2 and longer sequences.

    The operation is tokenwise. It is not a second spatial Transformer.
    Irregular intervals change the weighted temporal statistics but do not
    change the spatial token count.
    """

    def __init__(self, visual_dim: int) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(5 * visual_dim),
            nn.Linear(5 * visual_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )
        self.interval_projection = nn.Sequential(
            nn.Linear(1, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )

    def forward(
        self,
        frames: Tensor,
        time_intervals: Tensor | None = None,
    ) -> Tensor:
        if frames.ndim != 4:
            raise ValueError("frames must be [B,T,N,D]")
        batch, times, tokens, dim = frames.shape
        if times < 2 or dim != self.visual_dim:
            raise ValueError(
                f"frames must have T>=2 and D={self.visual_dim}; got {tuple(frames.shape)}"
            )
        deltas = frames[:, 1:] - frames[:, :-1]
        if time_intervals is None:
            weights = torch.ones(
                batch,
                times - 1,
                device=frames.device,
                dtype=frames.dtype,
            )
        else:
            if time_intervals.ndim == 1:
                time_intervals = time_intervals.unsqueeze(0).expand(batch, -1)
            if tuple(time_intervals.shape) != (batch, times - 1):
                raise ValueError("time_intervals must be [B,T-1] or [T-1]")
            weights = time_intervals.to(device=frames.device, dtype=frames.dtype)
            weights = weights.clamp_min(1e-6)
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
        weighted = weights[:, :, None, None]
        normalizer = weighted.sum(dim=1).clamp_min(1e-6)
        signed_delta = (deltas * weighted).sum(dim=1) / normalizer
        magnitude = (deltas.abs() * weighted).sum(dim=1) / normalizer
        interaction = (
            (frames[:, 1:] * frames[:, :-1] * weighted).sum(dim=1) / normalizer
        )
        context = frames.mean(dim=1)
        long_delta = frames[:, -1] - frames[:, 0]
        descriptor = torch.cat(
            (context, signed_delta, magnitude, interaction, long_delta),
            dim=-1,
        )
        projected = self.projection(descriptor)
        absolute_interval = time_intervals
        if absolute_interval is None:
            absolute_interval = torch.ones(
                batch,
                times - 1,
                device=frames.device,
                dtype=frames.dtype,
            )
        interval_feature = torch.log1p(
            absolute_interval.to(device=frames.device, dtype=frames.dtype).mean(dim=1, keepdim=True)
        )
        return projected + self.interval_projection(interval_feature).unsqueeze(1)


class TemporalTokenPyramid(nn.Module):
    """Compress adapted temporal tokens to dense and context scales."""

    def __init__(self, cfg: QCPRHRGConfig) -> None:
        super().__init__()
        if cfg.token_dim != cfg.retrieval_dim:
            raise ValueError("the first HRG implementation requires token_dim=retrieval_dim")
        self.cfg = cfg
        self.dense_projection = nn.Linear(cfg.visual_dim, cfg.token_dim)
        self.context_projection = nn.Linear(cfg.visual_dim, cfg.token_dim)
        self.slot_queries = nn.Parameter(torch.zeros(1, cfg.slot_count, cfg.token_dim))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        if cfg.token_dim % cfg.slot_heads:
            raise ValueError("token_dim must divide slot_heads")
        self.slot_attention = nn.MultiheadAttention(
            cfg.token_dim,
            cfg.slot_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.slot_projection = nn.Linear(cfg.token_dim, cfg.retrieval_dim)

    @staticmethod
    def _grid_shape(
        token_count: int,
        grid_shape: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if grid_shape is not None:
            if grid_shape[0] * grid_shape[1] != token_count:
                raise ValueError("grid_shape does not match native token count")
            return int(grid_shape[0]), int(grid_shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError("non-square native token count requires explicit grid_shape")
        return side, side

    def forward(
        self,
        adapted_tokens: Tensor,
        grid_shape: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, int]]:
        if adapted_tokens.ndim != 3:
            raise ValueError("adapted_tokens must be [B,N,D]")
        batch, token_count, dim = adapted_tokens.shape
        if dim != self.cfg.visual_dim:
            raise ValueError("adapted token dimension mismatch")
        height, width = self._grid_shape(token_count, grid_shape)
        image = adapted_tokens.transpose(1, 2).reshape(batch, dim, height, width)
        dense_image = F.adaptive_avg_pool2d(
            image, (self.cfg.token_grid, self.cfg.token_grid)
        )
        context_image = F.adaptive_avg_pool2d(
            image, (self.cfg.context_grid, self.cfg.context_grid)
        )
        dense = self.dense_projection(
            dense_image.flatten(2).transpose(1, 2)
        )
        context = self.context_projection(
            context_image.flatten(2).transpose(1, 2)
        )
        pyramid = torch.cat((dense, context), dim=1)
        queries = self.slot_queries.expand(batch, -1, -1)
        slots, _ = self.slot_attention(
            queries,
            pyramid,
            pyramid,
            need_weights=False,
        )
        slots = F.normalize(self.slot_projection(slots), dim=-1)
        return dense, context, slots, (self.cfg.token_grid, self.cfg.token_grid)


class QCPRHierarchicalRetriever(nn.Module):
    """Global ANN vector + slot/late/evidence candidate scoring.

    The global path is always computed for the complete gallery. The late and
    evidence paths are intended for a Top-K candidate tensor only. The three
    residual score gates start at zero, making the initial score exactly the
    global score while retaining the additional branches for controlled
    training.
    """

    def __init__(self, cfg: QCPRHRGConfig = QCPRHRGConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        pair_cfg = SinglePassConfig(
            visual_dim=cfg.visual_dim,
            text_dim=cfg.retrieval_dim,
            retrieval_dim=cfg.retrieval_dim,
            grid_size=cfg.token_grid,
            token_adapter_layers=cfg.token_adapter_layers,
            token_bottleneck_ratio=cfg.token_bottleneck_ratio,
            pair_layers=cfg.pair_layers,
            pair_heads=cfg.pair_heads,
            ffn_ratio=cfg.ffn_ratio,
            dropout=cfg.dropout,
            text_adapter_layers=0,
        )
        self.temporal_descriptor = TemporalDescriptorBuilder(cfg.visual_dim)
        self.pair_adapter = DeepResidualPairAdapter(pair_cfg)
        self.token_pyramid = TemporalTokenPyramid(cfg)
        self.late_gate = nn.Parameter(torch.zeros(()))
        self.slot_gate = nn.Parameter(torch.zeros(()))
        self.evidence_gate = nn.Parameter(torch.zeros(()))

    def encode_visual(
        self,
        frames: Tensor,
        *,
        grid_shape: tuple[int, int] | None = None,
        time_intervals: Tensor | None = None,
    ) -> HRGVisualEncoding:
        native = self.temporal_descriptor(frames, time_intervals)
        pair_encoding = self.pair_adapter(native)
        dense, context, slots, evidence_grid = self.token_pyramid(
            pair_encoding.adapted_dense_tokens,
            grid_shape,
        )
        global_vector = pair_encoding.pair_search_vector
        metadata = {
            **pair_encoding.metadata,
            "input_time_count": int(frames.shape[1]),
            "native_grid": list(TemporalTokenPyramid._grid_shape(frames.shape[2], grid_shape)),
            "dense_change_token_count": int(dense.shape[1]),
            "context_token_count": int(context.shape[1]),
            "pyramid_token_count": int(dense.shape[1] + context.shape[1]),
            "change_slot_count": int(slots.shape[1]),
            "evidence_grid": list(evidence_grid),
            "global_indexable": True,
            "late_interaction_requires_candidates": True,
            "full_gallery_cross_attention": False,
        }
        return HRGVisualEncoding(
            global_pair_vector=global_vector,
            change_slot_vectors=slots,
            dense_change_tokens=dense,
            pyramid_tokens=torch.cat((dense, context), dim=1),
            metadata=metadata,
        )

    @staticmethod
    def _validate_text(
        query_global: Tensor,
        query_tokens: Tensor,
        valid_token_mask: Tensor,
        dim: int,
    ) -> None:
        if query_global.ndim != 2 or query_global.shape[-1] != dim:
            raise ValueError("query_global must be [Q,D]")
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != dim:
            raise ValueError("query_tokens must be [Q,L,D]")
        if valid_token_mask.shape != query_tokens.shape[:2]:
            raise ValueError("valid_token_mask must be [Q,L]")
        if not valid_token_mask.bool().any(dim=1).all():
            raise ValueError("every query needs at least one valid content token")

    def _token_similarities(
        self,
        query_tokens: Tensor,
        visual_tokens: Tensor,
    ) -> Tensor:
        query = F.normalize(query_tokens, dim=-1)
        visual = F.normalize(visual_tokens, dim=-1)
        return torch.einsum("qld,pmd->qplm", query, visual)

    def candidate_scores(
        self,
        query_global: Tensor,
        query_tokens: Tensor,
        valid_token_mask: Tensor,
        visual: HRGVisualEncoding,
    ) -> HRGRetrievalOutput:
        self._validate_text(
            query_global,
            query_tokens,
            valid_token_mask,
            self.cfg.retrieval_dim,
        )
        global_scores = F.normalize(query_global, dim=-1) @ visual.global_pair_vector.T
        slot_sim = torch.einsum(
            "qd,pkd->qpk",
            F.normalize(query_global, dim=-1),
            visual.change_slot_vectors,
        )
        slot_scores = slot_sim.amax(dim=-1)
        similarities = self._token_similarities(
            query_tokens,
            visual.pyramid_tokens,
        )
        valid = valid_token_mask.bool()[:, None, :, None]
        best_for_text_token = similarities.amax(dim=-1)
        masked_best = best_for_text_token.masked_fill(~valid[:, :, :, 0], 0.0)
        denominator = valid_token_mask.sum(dim=-1).clamp_min(1).to(masked_best.dtype)
        late_scores = masked_best.sum(dim=-1) / denominator[:, None]
        evidence_similarities = self._token_similarities(
            query_tokens,
            visual.dense_change_tokens,
        )
        evidence_logits = evidence_similarities.masked_fill(~valid, -torch.inf).amax(dim=2)
        topk = min(self.cfg.evidence_topk, evidence_logits.shape[-1])
        top_values = evidence_logits.topk(topk, dim=-1).values
        evidence_scores = torch.logsumexp(top_values, dim=-1) - torch.log(
            torch.tensor(float(topk), device=top_values.device, dtype=top_values.dtype)
        )
        evidence_probabilities = evidence_logits.sigmoid()
        scores = (
            global_scores
            + self.slot_gate * slot_scores
            + self.late_gate * late_scores
            + self.evidence_gate * evidence_scores
        )
        return HRGRetrievalOutput(
            scores=scores,
            global_scores=global_scores,
            slot_scores=slot_scores,
            late_scores=late_scores,
            evidence_scores=evidence_scores,
            evidence_logits=evidence_logits,
            evidence_probabilities=evidence_probabilities,
            visual=visual,
        )

    def rerank_candidates(
        self,
        query_global: Tensor,
        query_tokens: Tensor,
        valid_token_mask: Tensor,
        gallery_visual: HRGVisualEncoding,
        candidate_indices: Tensor,
    ) -> HRGRetrievalOutput:
        """Score per-query candidates without touching the full gallery tokens."""
        if candidate_indices.ndim == 1:
            candidate_indices = candidate_indices.unsqueeze(0).expand(
                query_global.shape[0], -1
            )
        if candidate_indices.ndim != 2:
            raise ValueError("candidate_indices must be [Q,K] or [K]")
        if candidate_indices.shape[0] != query_global.shape[0]:
            raise ValueError("candidate rows must match query rows")
        if candidate_indices.numel() and int(candidate_indices.max()) >= gallery_visual.global_pair_vector.shape[0]:
            raise ValueError("candidate index exceeds gallery size")
        selected = HRGVisualEncoding(
            global_pair_vector=gallery_visual.global_pair_vector[candidate_indices],
            change_slot_vectors=gallery_visual.change_slot_vectors[candidate_indices],
            dense_change_tokens=gallery_visual.dense_change_tokens[candidate_indices],
            pyramid_tokens=gallery_visual.pyramid_tokens[candidate_indices],
            metadata={
                **gallery_visual.metadata,
                "candidate_count": int(candidate_indices.shape[1]),
            },
        )
        self._validate_text(
            query_global,
            query_tokens,
            valid_token_mask,
            self.cfg.retrieval_dim,
        )
        query = F.normalize(query_global, dim=-1)
        global_scores = (query[:, None, :] * selected.global_pair_vector).sum(-1)
        slot_sim = torch.einsum(
            "qd,qksd->qks",
            query,
            selected.change_slot_vectors,
        )
        slot_scores = slot_sim.amax(dim=-1)
        similarities = torch.einsum(
            "qld,qkmd->qklm",
            F.normalize(query_tokens, dim=-1),
            F.normalize(selected.pyramid_tokens, dim=-1),
        )
        valid = valid_token_mask.bool()[:, None, :, None]
        best_for_text_token = similarities.amax(dim=-1)
        masked_best = best_for_text_token.masked_fill(~valid[:, :, :, 0], 0.0)
        denominator = valid_token_mask.sum(dim=-1).clamp_min(1).to(masked_best.dtype)
        late_scores = masked_best.sum(dim=-1) / denominator[:, None]
        evidence_similarities = torch.einsum(
            "qld,qkmd->qklm",
            F.normalize(query_tokens, dim=-1),
            F.normalize(selected.dense_change_tokens, dim=-1),
        )
        evidence_logits = evidence_similarities.masked_fill(~valid, -torch.inf).amax(dim=2)
        topk = min(self.cfg.evidence_topk, evidence_logits.shape[-1])
        top_values = evidence_logits.topk(topk, dim=-1).values
        evidence_scores = torch.logsumexp(top_values, dim=-1) - torch.log(
            torch.tensor(float(topk), device=top_values.device, dtype=top_values.dtype)
        )
        evidence_probabilities = evidence_logits.sigmoid()
        scores = (
            global_scores
            + self.slot_gate * slot_scores
            + self.late_gate * late_scores
            + self.evidence_gate * evidence_scores
        )
        return HRGRetrievalOutput(
            scores=scores,
            global_scores=global_scores,
            slot_scores=slot_scores,
            late_scores=late_scores,
            evidence_scores=evidence_scores,
            evidence_logits=evidence_logits,
            evidence_probabilities=evidence_probabilities,
            visual=selected,
        )
