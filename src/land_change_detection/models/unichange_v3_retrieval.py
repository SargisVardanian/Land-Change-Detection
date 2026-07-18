from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.models.qcpr_v3 import (
    QCPRV3AlignedMaskOutput,
    QCPRV3GenericGrounding,
    QCPRV3ScoreOutput,
)


@dataclass(frozen=True)
class UniChangeV3Output:
    pair_embedding: Tensor
    base_text_embedding: Tensor
    text_embedding: Tensor
    text_token_embeddings: Tensor
    text_attention_mask: Tensor
    text_content_mask: Tensor
    per_time_tokens: Tensor
    scores: QCPRV3ScoreOutput
    visual_metadata: dict[str, Any]


@dataclass(frozen=True)
class UniChangeV3GlobalOutput:
    pair_embedding: Tensor
    base_text_embedding: Tensor
    text_embedding: Tensor
    text_token_embeddings: Tensor
    text_attention_mask: Tensor
    text_content_mask: Tensor
    per_time_tokens: Tensor
    global_score: Tensor
    visual_metadata: dict[str, Any]


@dataclass(frozen=True)
class UniChangeV3AlignedMaskOutput:
    pair_embedding: Tensor
    base_text_embedding: Tensor
    text_embedding: Tensor
    text_token_embeddings: Tensor
    text_attention_mask: Tensor
    text_content_mask: Tensor
    per_time_tokens: Tensor
    masks: QCPRV3AlignedMaskOutput
    visual_metadata: dict[str, Any]


class UniChangeV3RetrievalModel(nn.Module):
    """Global v1-compatible dual encoder plus generic v3 grounding model."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        temporal_encoder: nn.Module,
        text_encoder: nn.Module,
        retrieval_head: nn.Module,
        grounder: QCPRV3GenericGrounding,
        text_adapter: nn.Module | None = None,
        grounding_backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.temporal_encoder = temporal_encoder
        self.text_encoder = text_encoder
        self.retrieval_head = retrieval_head
        self.text_adapter = text_adapter
        self.grounding_backbone = grounding_backbone
        self.grounder = grounder
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        image_encoder = getattr(self.visual_encoder, "image_encoder", self.visual_encoder)
        for parameter in image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()
        self.text_encoder.eval()
        if self.grounding_backbone is not None:
            self.grounding_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual_encoder.eval()
        self.text_encoder.eval()
        if self.grounding_backbone is not None:
            self.grounding_backbone.eval()
        return self

    def encode_pairs(self, images: Tensor, temporal_valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor, dict[str, Any]]:
        with torch.no_grad():
            visual = self.visual_encoder(images)
        temporal = self.temporal_encoder(visual.features, temporal_valid_mask=temporal_valid_mask)
        pair = F.normalize(self.retrieval_head(temporal.pair_embedding).pair_embedding, dim=-1)
        # Global retrieval remains UniverSat. A separately selected frozen
        # dense VLM may provide the ordered grounding field without changing
        # candidate generation.
        grounding_features = visual.features
        metadata = dict(visual.metadata)
        if self.grounding_backbone is not None:
            with torch.no_grad():
                grounding_features = self.grounding_backbone.encode_images(images)
            metadata["grounding_backbone"] = self.grounding_backbone.provenance()
        return pair, grounding_features, metadata

    def encode_texts(self, captions: list[str]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            features = self.text_encoder(captions, role="query")
        if not isinstance(features, TextFeatures) and not hasattr(features, "global_embedding"):
            raise TypeError("text encoder must return TextFeatures-compatible output")
        base_embedding = F.normalize(features.global_embedding, dim=-1)
        global_embedding = base_embedding
        if self.text_adapter is not None:
            global_embedding = self.text_adapter(global_embedding)
        token_embeddings = features.token_embeddings
        attention_mask = features.attention_mask.bool()
        content_mask = features.content_token_mask.bool()
        if self.grounding_backbone is not None:
            grounding_text = self.grounding_backbone.encode_texts(
                captions,
                device=base_embedding.device,
            )
            token_embeddings = grounding_text.token_embeddings
            attention_mask = grounding_text.attention_mask
            content_mask = grounding_text.content_mask
        return (
            base_embedding,
            F.normalize(global_embedding, dim=-1),
            token_embeddings,
            attention_mask,
            content_mask,
        )

    def score_encoded(
        self,
        pair_embedding: Tensor,
        per_time_tokens: Tensor,
        text_embedding: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
        *,
        decode_mask: bool = True,
    ) -> QCPRV3ScoreOutput:
        return self.grounder.score_query_pair_chunks(
            text_embedding,
            text_tokens,
            text_attention_mask,
            pair_embedding,
            per_time_tokens,
            text_content_mask=text_content_mask,
            decode_mask=decode_mask,
        )

    def forward_global(
        self,
        images: Tensor,
        captions: list[str],
        temporal_valid_mask: Tensor | None = None,
    ) -> UniChangeV3GlobalOutput:
        pair, per_time, metadata = self.encode_pairs(images, temporal_valid_mask)
        base_text, text, tokens, attention, content = self.encode_texts(captions)
        base_text = base_text.to(pair.device)
        text = text.to(pair.device)
        tokens = tokens.to(pair.device)
        attention = attention.to(pair.device)
        content = content.to(pair.device)
        return UniChangeV3GlobalOutput(
            pair,
            base_text,
            text,
            tokens,
            attention,
            content,
            per_time,
            text @ pair.T,
            metadata,
        )

    def forward_aligned_masks(
        self,
        images: Tensor,
        captions: list[str],
        query_to_pair: Tensor,
        temporal_valid_mask: Tensor | None = None,
    ) -> UniChangeV3AlignedMaskOutput:
        """Supervised M0 forward with one decoded mask per directional query."""
        pair, per_time, metadata = self.encode_pairs(images, temporal_valid_mask)
        base_text, text, tokens, attention, content = self.encode_texts(captions)
        base_text = base_text.to(pair.device)
        text = text.to(pair.device)
        tokens = tokens.to(pair.device)
        attention = attention.to(pair.device)
        content = content.to(pair.device)
        query_to_pair = query_to_pair.to(pair.device)
        masks = self.grounder.score_aligned_query_pairs(
            text,
            tokens,
            attention,
            per_time,
            query_to_pair,
        )
        return UniChangeV3AlignedMaskOutput(
            pair,
            base_text,
            text,
            tokens,
            attention,
            content,
            per_time,
            masks,
            metadata,
        )

    def forward(
        self,
        images: Tensor,
        captions: list[str],
        caption_to_pair: Tensor | None = None,
        temporal_valid_mask: Tensor | None = None,
        *,
        decode_mask: bool = True,
    ) -> UniChangeV3Output:
        pair, per_time, metadata = self.encode_pairs(images, temporal_valid_mask)
        base_text, text, tokens, attention, content = self.encode_texts(captions)
        tokens = tokens.to(pair.device)
        attention = attention.to(pair.device)
        content = content.to(pair.device)
        text = text.to(pair.device)
        base_text = base_text.to(pair.device)
        scores = self.score_encoded(
            pair, per_time, text, tokens, attention, content,
            decode_mask=decode_mask,
        )
        return UniChangeV3Output(pair, base_text, text, tokens, attention, content, per_time, scores, metadata)
