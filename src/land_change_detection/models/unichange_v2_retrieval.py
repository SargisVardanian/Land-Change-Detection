from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.models.retrieval_heads import RetrievalProjectionHead
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder


@dataclass(frozen=True)
class UniChangeV2RetrievalOutput:
    pair_embedding: Tensor
    text_embedding: Tensor
    logits: Tensor
    visual_metadata: dict[str, Any]


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
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.temporal_encoder = temporal_encoder
        self.text_encoder = text_encoder
        self.retrieval_head = retrieval_head
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        for parameter in self.visual_encoder.image_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()
        self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual_encoder.eval()
        self.text_encoder.eval()
        return self

    def encode_pairs(self, images: Tensor, temporal_valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Any]]:
        visual = self.visual_encoder(images)
        temporal = self.temporal_encoder(visual.features, temporal_valid_mask=temporal_valid_mask)
        projected = self.retrieval_head(temporal.pair_embedding)
        return projected.pair_embedding, visual.metadata

    def encode_texts(self, captions: list[str]) -> Tensor:
        with torch.no_grad():
            features = self.text_encoder(captions, role="document")
        if not isinstance(features, TextFeatures) and not hasattr(features, "global_embedding"):
            raise TypeError("text_encoder must return an object with global_embedding")
        return features.global_embedding.detach()

    def forward(
        self,
        images: Tensor,
        captions: list[str],
        caption_to_pair: Tensor,
        temporal_valid_mask: Tensor | None = None,
    ) -> UniChangeV2RetrievalOutput:
        pair_embedding, metadata = self.encode_pairs(images, temporal_valid_mask=temporal_valid_mask)
        text_embedding = self.encode_texts(captions).to(pair_embedding.device)
        logits = pair_embedding @ text_embedding.T
        return UniChangeV2RetrievalOutput(
            pair_embedding=pair_embedding,
            text_embedding=text_embedding,
            logits=logits,
            visual_metadata=metadata,
        )
