from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead, TextEmbeddingAdapter
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder
from land_change_detection.models.qcpr import QCPRPatchReranker, temporal_patch_descriptor


@dataclass(frozen=True)
class UniChangeV2RetrievalOutput:
    pair_embedding: Tensor
    text_embedding: Tensor
    teacher_text_embedding: Tensor
    logits: Tensor
    visual_metadata: dict[str, Any]
    patch_tokens: Tensor | None = None
    global_scores: Tensor | None = None
    local_scores: Tensor | None = None
    final_scores: Tensor | None = None
    query_mask_logits: Tensor | None = None
    mask_query_embeddings: Tensor | None = None
    temporal_explanation_logits: Tensor | None = None
    score_mode: str = "global"


class UniChangeV2RetrievalModel(nn.Module):
    """Minimal UniChange v2 Stage-1 retrieval model.

    This model intentionally covers only the first cluster-readiness objective:
    frozen per-frame UniverSat + temporal change encoder + frozen Jina text
    embeddings + CLIP-like text-to-pair retrieval.
    """

    def __init__(
        self,
        visual_encoder: SequenceUniverSatEncoder,
        temporal_encoder: TemporalChangeEncoder,
        text_encoder: nn.Module,
        retrieval_head: RetrievalProjectionHead,
        text_adapter: TextEmbeddingAdapter | None = None,
        patch_reranker: QCPRPatchReranker | None = None,
        temporal_explanation_head: nn.Module | None = None,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.temporal_encoder = temporal_encoder
        self.text_encoder = text_encoder
        self.retrieval_head = retrieval_head
        self.text_adapter = text_adapter
        self.patch_reranker = patch_reranker
        self.temporal_explanation_head = temporal_explanation_head
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        for parameter in self.visual_encoder.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()
        self.text_encoder.eval()

    @property
    def patch_projector(self) -> nn.Module | None:
        return self.patch_reranker.patch_projector if self.patch_reranker is not None else None

    @property
    def query_mask_head(self) -> nn.Module | None:
        return self.patch_reranker.query_mask_head if self.patch_reranker is not None else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual_encoder.eval()
        self.text_encoder.eval()
        return self

    def encode_pairs(self, images: Tensor, temporal_valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Any]]:
        pair_embedding, _, metadata = self.encode_pair_features(images, temporal_valid_mask=temporal_valid_mask)
        return pair_embedding, metadata

    def encode_pair_features(self, images: Tensor, temporal_valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        visual = self.visual_encoder(images)
        temporal = self.temporal_encoder(visual.features, temporal_valid_mask=temporal_valid_mask)
        projected = self.retrieval_head(temporal.pair_embedding)
        patches = self.patch_reranker.project_patches(temporal.change_tokens) if self.patch_reranker is not None else None
        return projected.pair_embedding, patches, visual.metadata

    def encode_pair_explanations(self, images: Tensor, temporal_valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None, dict[str, Any]]:
        """Encode retrieval patches plus optional appeared/disappeared/changed logits."""
        visual = self.visual_encoder(images)
        temporal = self.temporal_encoder(visual.features, temporal_valid_mask=temporal_valid_mask)
        projected = self.retrieval_head(temporal.pair_embedding)
        patches = self.patch_reranker.project_patches(temporal.change_tokens) if self.patch_reranker is not None else None
        direction_logits = None
        if self.temporal_explanation_head is not None:
            if temporal.per_time_tokens.shape[1] != 2:
                raise ValueError("Temporal explanation channels require exactly T1 and T2")
            direction_logits = self.temporal_explanation_head(temporal_patch_descriptor(temporal.per_time_tokens))
        return projected.pair_embedding, patches, direction_logits, visual.metadata

    def encode_texts(self, captions: list[str], *, return_teacher: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        with torch.no_grad():
            features = self.text_encoder(captions, role="query")
        if not isinstance(features, TextFeatures) and not hasattr(features, "global_embedding"):
            raise TypeError("text_encoder must return an object with global_embedding")
        teacher_embeddings = features.global_embedding.detach()
        embeddings = teacher_embeddings
        if self.text_adapter is not None:
            embeddings = self.text_adapter(embeddings)
        if return_teacher:
            return embeddings, teacher_embeddings
        return embeddings

    def forward(
        self,
        images: Tensor,
        captions: list[str],
        caption_to_pair: Tensor,
        temporal_valid_mask: Tensor | None = None,
    ) -> UniChangeV2RetrievalOutput:
        pair_embedding, patch_tokens, temporal_explanation_logits, metadata = self.encode_pair_explanations(
            images, temporal_valid_mask=temporal_valid_mask
        )
        text_embedding, teacher_text_embedding = self.encode_texts(captions, return_teacher=True)
        text_embedding = text_embedding.to(pair_embedding.device)
        teacher_text_embedding = teacher_text_embedding.to(pair_embedding.device)
        global_scores = text_embedding @ pair_embedding.T
        local_scores = None
        final_scores = global_scores
        query_mask_logits = None
        mask_query_embeddings = None
        score_mode = "global"
        if self.patch_reranker is not None and patch_tokens is not None:
            reranked = self.patch_reranker.score(text_embedding, pair_embedding, patch_tokens)
            global_scores = reranked["global_score"]  # type: ignore[assignment]
            local_scores = reranked["local_score"]  # type: ignore[assignment]
            final_scores = reranked["final_score"]  # type: ignore[assignment]
            query_mask_logits = reranked["query_mask_logits"]  # type: ignore[assignment]
            mask_query_embeddings = reranked["mask_query_embeddings"]  # type: ignore[assignment]
            score_mode = str(reranked["score_mode"])
        logits = final_scores.T
        return UniChangeV2RetrievalOutput(
            pair_embedding=pair_embedding,
            text_embedding=text_embedding,
            teacher_text_embedding=teacher_text_embedding,
            logits=logits,
            visual_metadata=metadata,
            patch_tokens=patch_tokens,
            global_scores=global_scores,
            local_scores=local_scores,
            final_scores=final_scores,
            query_mask_logits=query_mask_logits,
            mask_query_embeddings=mask_query_embeddings,
            temporal_explanation_logits=temporal_explanation_logits,
            score_mode=score_mode,
        )
