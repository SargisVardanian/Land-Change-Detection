from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..backbones.siglip2 import ImageEncoding, Siglip2Backbone, TextEncoding
from ..config.schema import Siglip2TemporalConfig
from ..contracts import validate_feature_contract
from .evidence import EvidenceBottleneck, EvidenceOutput
from .relevance import UnifiedRelevanceModel
from .temporal import TemporalAdapterOutput, TemporalTransformerAdapter


@dataclass
class RetrievalForwardOutput:
    score_matrix: Tensor
    global_score_matrix: Tensor
    evidence_diagnostic_score_matrix: Tensor
    text_embedding: Tensor
    pair_cls: Tensor
    pair_for_query: Tensor
    evidence: EvidenceOutput
    temporal: TemporalAdapterOutput


class Siglip2TemporalRetrievalModel(nn.Module):
    """Minimal SigLIP-2 temporal retriever with a causal evidence score."""

    def __init__(
        self,
        backbone: Siglip2Backbone | None,
        config: Siglip2TemporalConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = (config or Siglip2TemporalConfig()).validate()
        self.backbone = backbone
        self.temporal_adapter = TemporalTransformerAdapter(self.config)
        self.evidence_bottleneck = EvidenceBottleneck(self.config)
        self.relevance_model = UnifiedRelevanceModel()
        self.log_temperature = nn.Parameter(
            torch.tensor(self.config.retrieval_temperature).log()
        )
        if self.config.retrieval_score_mode == "final_v1_primary":
            self.evidence_bottleneck.requires_grad_(False)
            self.relevance_model.requires_grad_(False)

    @property
    def retrieval_temperature(self) -> Tensor:
        return self.log_temperature.clamp(min=-5.0, max=2.0).exp()

    def unified_score_from_evidence(
        self,
        text_embedding: Tensor,
        pair_cls: Tensor,
        change_tokens: Tensor,
        evidence_gate: Tensor,
        evidence_vector: Tensor,
        token_evidence_score: Tensor | None = None,
    ) -> Tensor:
        """Compute the canonical scalar score for an arbitrary evidence vector."""

        if (
            text_embedding.ndim != 2
            or pair_cls.ndim != 2
            or change_tokens.ndim != 3
            or evidence_vector.ndim != 3
        ):
            raise ValueError("unified score inputs have invalid ranks")
        global_score = text_embedding @ pair_cls.transpose(0, 1)
        query_pair = F.normalize(
            pair_cls.unsqueeze(0) + evidence_gate * evidence_vector, dim=-1
        )
        pair_score = torch.einsum("qd,qpd->qp", text_embedding, query_pair)
        vector_evidence_score = pair_score - global_score
        if token_evidence_score is None:
            evidence_score = vector_evidence_score
        else:
            if token_evidence_score.shape != global_score.shape:
                raise ValueError("token evidence score must match [queries, pairs]")
            # Keep both causal routes in the one scalar relevance score.  The
            # token score is normalized log-mean-exp similarity from the same
            # evidence weights; the vector score preserves gradients through
            # the weighted visual representation.
            evidence_score = 0.5 * vector_evidence_score + 0.5 * token_evidence_score
        change_normalized = F.normalize(change_tokens, dim=-1)
        slot_score = torch.logsumexp(
            torch.einsum("qd,pkd->qpk", text_embedding, change_normalized),
            dim=-1,
        )
        unified_score = self.relevance_model(
            global_score,
            evidence_gate * evidence_score,
            slot_score,
        )
        return unified_score / self.retrieval_temperature

    def forward_from_features(
        self,
        frame_tokens: Tensor,
        frame_embeddings: Tensor,
        text_tokens: Tensor,
        text_embeddings: Tensor,
        text_mask: Tensor,
        *,
        timestamps: Tensor | None = None,
        patch_valid_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        native_image_size: Tensor | None = None,
        processed_patch_grid: Tensor | None = None,
        transform_hash: str | None = None,
        frame_ids: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        gsd: Tensor | None = None,
        metadata_missing: Tensor | None = None,
        token_coordinates: Tensor | None = None,
        force_region_reduction: bool = False,
    ) -> RetrievalForwardOutput:
        validate_feature_contract(
            frame_tokens,
            frame_embeddings,
            text_tokens,
            text_embeddings,
            text_mask,
            hidden_size=self.config.hidden_size,
            expected_patch_tokens=self.config.expected_patch_tokens,
        )
        temporal = self.temporal_adapter(
            frame_tokens,
            frame_embeddings,
            timestamps=timestamps,
            patch_valid_mask=patch_valid_mask,
            spatial_shapes=spatial_shapes,
            native_image_size=native_image_size,
            processed_patch_grid=processed_patch_grid,
            transform_hash=transform_hash,
            frame_ids=frame_ids,
            sensor_ids=sensor_ids,
            gsd=gsd,
            metadata_missing=metadata_missing,
            token_coordinates=token_coordinates,
            force_region_reduction=force_region_reduction,
        )
        text_embedding = F.normalize(text_embeddings, dim=-1)
        pair = temporal.pair_cls
        evidence = self.evidence_bottleneck(
            text_tokens,
            text_mask,
            temporal.temporal_patch_tokens,
            frame_count=temporal.frame_count,
            patch_count=temporal.patch_count,
            patch_valid_mask=temporal.patch_valid_mask,
            spatial_shapes=temporal.spatial_shapes,
            native_patch_valid_mask=temporal.native_patch_valid_mask,
            native_spatial_shapes=temporal.native_spatial_shapes,
            region_assignment=temporal.region_assignment,
        )
        pair_for_query = F.normalize(
            pair.unsqueeze(0) + evidence.evidence_gate * evidence.evidence_vector,
            dim=-1,
        )
        global_score = text_embedding @ pair.transpose(0, 1)
        evidence_diagnostic_score = self.unified_score_from_evidence(
            text_embedding,
            pair,
            temporal.change_tokens,
            evidence.evidence_gate,
            evidence.evidence_vector,
            evidence.evidence_score,
        )
        primary_score = global_score / self.retrieval_temperature
        score = (
            evidence_diagnostic_score
            if self.config.retrieval_score_mode == "evidence_mechanism_ablation"
            else primary_score
        )
        return RetrievalForwardOutput(
            score,
            global_score,
            evidence_diagnostic_score,
            text_embedding,
            pair,
            pair_for_query,
            evidence,
            temporal,
        )

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        pixel_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        content_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
        native_image_size: Tensor | None = None,
        transform_hash: str | None = None,
        frame_ids: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        gsd: Tensor | None = None,
        metadata_missing: Tensor | None = None,
        token_coordinates: Tensor | None = None,
    ) -> RetrievalForwardOutput:
        if self.backbone is None:
            raise RuntimeError("raw-input forward requires a Siglip2Backbone")
        image: ImageEncoding = self.backbone.encode_images(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            native_image_size=native_image_size,
            transform_hash=transform_hash,
        )
        text: TextEncoding = self.backbone.encode_text(
            input_ids,
            attention_mask,
            content_mask=content_mask,
        )
        return self.forward_from_features(
            image.patch_tokens,
            image.pooled_embedding,
            text.token_embeddings,
            text.pooled_embedding,
            text.attention_mask,
            timestamps=timestamps,
            patch_valid_mask=image.patch_valid_mask,
            spatial_shapes=image.spatial_shapes,
            native_image_size=image.native_image_size,
            processed_patch_grid=image.processed_patch_grid,
            transform_hash=image.transform_hash,
            frame_ids=frame_ids,
            sensor_ids=sensor_ids,
            gsd=gsd,
            metadata_missing=metadata_missing,
            token_coordinates=(
                image.token_coordinates
                if token_coordinates is None
                else token_coordinates
            ),
            force_region_reduction=image.force_region_reduction,
        )

    def trainable_parameter_report(self) -> dict[str, dict[str, int]]:
        report = {}
        for name, module in (
            ("temporal_adapter", self.temporal_adapter),
            ("evidence_bottleneck", self.evidence_bottleneck),
            ("relevance_model", self.relevance_model),
            ("retrieval_temperature", nn.ParameterList([self.log_temperature])),
        ):
            ps = list(module.parameters())
            report[name] = {
                "parameter_count": sum(p.numel() for p in ps),
                "trainable_count": sum(p.numel() for p in ps if p.requires_grad),
            }
        if self.backbone is not None:
            for name, module in (
                ("siglip2_vision_backbone", self.backbone.vision_model),
                ("siglip2_text_backbone", self.backbone.text_model),
            ):
                ps = list(module.parameters())
                report[name] = {
                    "parameter_count": sum(p.numel() for p in ps),
                    "trainable_count": sum(p.numel() for p in ps if p.requires_grad),
                }
        return report
