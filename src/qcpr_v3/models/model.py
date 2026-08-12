"""Canonical QCPR v3 temporal retrieval model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.schema import QCPRConfig, validate_config
from ..data.contracts import TemporalMetadata
from .evidence_bottleneck import EvidenceBottleneck, EvidenceOutput, PairwiseEvidenceOutput
from .jina_query_encoder import JinaQueryEncoder, QueryFeatures
from .relevance_model import RelevanceOutput, UnifiedRelevanceModel
from .temporal_adapter import TemporalAdapter, TemporalOutput


@dataclass(frozen=True)
class ModelScoreOutput:
    scores: Tensor
    relevance: RelevanceOutput
    evidence: PairwiseEvidenceOutput | None

    @property
    def evidence_weights(self) -> Tensor | None:
        return None if self.evidence is None else self.evidence.relevance_weights


class QCPRV3Model(nn.Module):
    """One shared temporal encoder, one query encoder and one scalar score."""

    def __init__(
        self,
        config: QCPRConfig | None = None,
        *,
        text_encoder: JinaQueryEncoder | None = None,
    ):
        super().__init__()
        self.config = validate_config(config or QCPRConfig())
        temporal = self.config.temporal
        text = self.config.text
        evidence = self.config.evidence
        self.temporal_adapter = TemporalAdapter(
            native_dim=temporal.native_dim,
            hidden_dim=temporal.hidden_dim,
            heads=temporal.heads,
            blocks=temporal.blocks,
            change_slots=temporal.change_slots,
            ff_ratio=temporal.ff_ratio,
            dropout=temporal.dropout,
            layer_scale_init=temporal.layer_scale_init,
            local_radius=temporal.local_radius,
            spatial_bands=temporal.spatial_rope_bands,
            temporal_bands=temporal.temporal_fourier_bands,
            max_frames=temporal.max_frames,
        )
        self.text_encoder = text_encoder or JinaQueryEncoder(
            input_dim=text.input_dim,
            hidden_dim=text.hidden_dim,
            adapter_layers=text.adapter_layers,
            heads=text.heads,
            ff_ratio=text.ff_ratio,
            dropout=text.dropout,
        )
        self.evidence_bottleneck = EvidenceBottleneck(
            hidden_dim=self.config.retrieval_dim,
            normalizer=evidence.sparse_normalizer,
            temperature=evidence.temperature,
        )
        self.relevance_model = UnifiedRelevanceModel(hidden_dim=self.config.retrieval_dim)

    def encode_visual(
        self,
        native_tokens: Tensor,
        coordinates: Tensor,
        temporal: TemporalMetadata,
        *,
        frame_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> TemporalOutput:
        return self.temporal_adapter(native_tokens, coordinates, temporal, frame_mask, token_mask)

    def encode_query(self, tokens: Tensor | list[str], mask: Tensor | None = None) -> QueryFeatures:
        return self.text_encoder(tokens, mask)

    def score(self, query: QueryFeatures, visual: TemporalOutput) -> ModelScoreOutput:
        query.validate()
        visual.validate()
        if query.text_tokens.shape[-1] != visual.sequence_cls.shape[-1]:
            raise ValueError("query and visual retrieval dimensions must match")
        text_cls = F.normalize(query.text_cls, dim=-1)
        sequence_cls = F.normalize(visual.sequence_cls, dim=-1)
        global_logit = text_cls @ sequence_cls.transpose(0, 1)
        change = F.normalize(visual.change_tokens, dim=-1)
        slot_logits = torch.einsum("qd,pkd->qpk", text_cls, change)
        slot_logit = torch.logsumexp(slot_logits, dim=-1) - torch.log(
            torch.tensor(float(change.shape[1]), device=slot_logits.device, dtype=slot_logits.dtype)
        )
        evidence: PairwiseEvidenceOutput | None = None
        if self.config.evidence.enabled:
            token_mask = visual.token_mask & visual.frame_mask.unsqueeze(-1)
            evidence = self.evidence_bottleneck.pairwise(
                query.text_tokens,
                query.text_mask,
                visual.dense_tokens,
                token_mask,
            )
            evidence_vector = F.normalize(evidence.evidence_vector, dim=-1)
            evidence_logit = (text_cls[:, None, :] * evidence_vector).sum(dim=-1)
            late_feature = evidence.late_score
        else:
            evidence_vector = torch.zeros(
                (query.text_cls.shape[0], visual.sequence_cls.shape[0], self.config.retrieval_dim),
                device=text_cls.device,
                dtype=text_cls.dtype,
            )
            evidence_logit = torch.zeros_like(global_logit)
            late_feature = torch.zeros_like(global_logit)
        relevance = self.relevance_model(
            global_logit,
            evidence_logit,
            slot_logit,
            late_feature,
            text_cls,
            sequence_cls,
            evidence_vector,
        )
        return ModelScoreOutput(relevance.score, relevance, evidence)

    def ann_scores(self, query: QueryFeatures, visual: TemporalOutput) -> Tensor:
        query.validate()
        visual.validate()
        return F.normalize(query.text_cls, dim=-1) @ F.normalize(visual.sequence_cls, dim=-1).transpose(0, 1)

    def rerank(self, query: QueryFeatures, visual: TemporalOutput, *, top_k: int = 50) -> tuple[Tensor, ModelScoreOutput]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query.text_cls.shape[0] != 1:
            raise ValueError("rerank currently requires one query at a time")
        stage_one = self.ann_scores(query, visual)
        k = min(top_k, visual.sequence_cls.shape[0])
        _, candidate_indices = stage_one.topk(k, dim=-1)
        selected = _select_visual(visual, candidate_indices)
        selected_scores = self.score_topk(query, selected)
        return candidate_indices, selected_scores

    def score_topk(self, query: QueryFeatures, visual: TemporalOutput) -> ModelScoreOutput:
        """Exact token-to-patch evidence score for one query and selected items."""
        if query.text_cls.shape[0] != 1:
            raise ValueError("score_topk requires one query at a time")
        query.validate()
        visual.validate()
        text_cls = F.normalize(query.text_cls, dim=-1)
        sequence_cls = F.normalize(visual.sequence_cls, dim=-1)
        global_logit = text_cls @ sequence_cls.transpose(0, 1)
        change = F.normalize(visual.change_tokens, dim=-1)
        slot_logits = torch.einsum("qd,pkd->qpk", text_cls, change)
        slot_logit = torch.logsumexp(slot_logits, dim=-1) - torch.log(
            torch.tensor(float(change.shape[1]), device=slot_logits.device, dtype=slot_logits.dtype)
        )
        token_mask = visual.token_mask & visual.frame_mask.unsqueeze(-1)
        exact: EvidenceOutput = self.evidence_bottleneck(
            query.text_tokens.expand(visual.dense_tokens.shape[0], -1, -1),
            query.text_mask.expand(visual.dense_tokens.shape[0], -1),
            visual.dense_tokens,
            token_mask,
        )
        evidence_vector = F.normalize(exact.evidence_vector, dim=-1)
        evidence_logit = (text_cls[:, None, :] * evidence_vector[None, :, :]).sum(dim=-1)
        if exact.token_similarity is None:
            raise RuntimeError("exact evidence output did not retain token similarities")
        late_feature = exact.token_similarity.amax(dim=-1).mean(dim=1).mean(dim=-1).unsqueeze(0)
        pairwise_evidence = PairwiseEvidenceOutput(
            evidence_vector.unsqueeze(0),
            exact.relevance_logits.unsqueeze(0),
            exact.relevance_weights.unsqueeze(0),
            late_feature,
        )
        relevance = self.relevance_model(
            global_logit,
            evidence_logit,
            slot_logit,
            late_feature,
            text_cls,
            sequence_cls,
            evidence_vector.unsqueeze(0),
        )
        return ModelScoreOutput(relevance.score, relevance, pairwise_evidence)

    def parameter_counts(self) -> dict[str, int]:
        all_count = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": all_count, "trainable": trainable}

    def parameter_group_counts(self) -> dict[str, dict[str, int]]:
        groups = {
            "temporal_adapter": self.temporal_adapter,
            "text_encoder": self.text_encoder,
            "evidence_bottleneck": self.evidence_bottleneck,
            "relevance_model": self.relevance_model,
        }
        return {
            name: {
                "total": sum(parameter.numel() for parameter in module.parameters()),
                "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
            }
            for name, module in groups.items()
        }

    def freeze_backbones(self) -> None:
        for parameter in self.temporal_adapter.parameters():
            parameter.requires_grad_(True)
        if self.text_encoder.base_encoder is not None:
            for parameter in self.text_encoder.base_encoder.parameters():
                parameter.requires_grad_(False)
        for module in (self.text_encoder.input_projection, self.text_encoder.adapter, self.text_encoder.output_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _select_visual(visual: TemporalOutput, indices: Tensor) -> TemporalOutput:
    if indices.ndim != 2:
        raise ValueError("indices must be [Q,K]")
    # Reranking evaluates one query at a time or a shared candidate set. For a
    # multi-query matrix, use the first row only to keep item identity explicit.
    item_indices = indices[0]
    temporal = visual.temporal
    sliced_temporal = TemporalMetadata(
        timestamps=temporal.timestamps.index_select(0, item_indices),
        delta_times=temporal.delta_times.index_select(0, item_indices),
        frame_ids=temporal.frame_ids.index_select(0, item_indices),
        sensor_ids=None if temporal.sensor_ids is None else temporal.sensor_ids.index_select(0, item_indices),
        gsd=None if temporal.gsd is None else temporal.gsd.index_select(0, item_indices),
        metadata_missing=None if temporal.metadata_missing is None else temporal.metadata_missing.index_select(0, item_indices),
    )
    return TemporalOutput(
        sequence_cls=visual.sequence_cls.index_select(0, item_indices),
        frame_cls=visual.frame_cls.index_select(0, item_indices),
        change_tokens=visual.change_tokens.index_select(0, item_indices),
        dense_tokens=visual.dense_tokens.index_select(0, item_indices),
        coordinates=visual.coordinates.index_select(0, item_indices),
        frame_mask=visual.frame_mask.index_select(0, item_indices),
        token_mask=visual.token_mask.index_select(0, item_indices),
        temporal=sliced_temporal,
    )
