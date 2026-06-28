from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from land_change_detection.backbones.jina_v5_text import JinaV5TextEncoder, TextFeatures
from land_change_detection.backbones.universat_backend import UniverSatJointBackend, VisualFeatureGrid


@dataclass(frozen=True)
class UniChangeOutput:
    global_pair_embedding: Tensor
    local_change_tokens: Tensor
    text_global_embedding: Tensor | None
    text_token_embeddings: Tensor | None
    event_embeddings: Tensor | None
    event_presence: Tensor | None
    event_mask_logits: Tensor | None
    event_masks: Tensor | None
    text_conditioned_mask: Tensor | None
    semantic_prediction: Tensor | None

    @property
    def event_presence_logits(self) -> Tensor | None:
        return self.event_presence

    @property
    def text_global_embeddings(self) -> Tensor | None:
        return self.text_global_embedding


@dataclass(frozen=True)
class UniChangeConfig:
    visual_dim: int = 768
    text_hidden_dim: int = 1024
    retrieval_dim: int = 512
    grid_height: int = 36
    grid_width: int = 36
    event_queries: int = 32
    predict_semantic: bool = True


class ChangeEventDecoder(nn.Module):
    def __init__(self, retrieval_dim: int = 512, event_queries: int = 32):
        super().__init__()
        self.event_queries = nn.Parameter(torch.randn(event_queries, retrieval_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=retrieval_dim,
            nhead=8,
            dim_feedforward=retrieval_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.presence_head = nn.Linear(retrieval_dim, 1)

    def forward(self, local_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = local_tokens.shape[0]
        queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        event_embeddings = F.normalize(self.decoder(queries, local_tokens), dim=-1)
        presence = self.presence_head(event_embeddings).squeeze(-1)
        mask_logits = event_embeddings @ local_tokens.transpose(1, 2)
        return event_embeddings, presence, mask_logits


class UniChangeModel(nn.Module):
    """Text-conditioned local semantic change retrieval model shell."""

    def __init__(
        self,
        visual_encoder: UniverSatJointBackend,
        text_encoder: JinaV5TextEncoder | None = None,
        config: UniChangeConfig | None = None,
    ):
        super().__init__()
        self.config = config or UniChangeConfig()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.event_decoder = ChangeEventDecoder(self.config.retrieval_dim, self.config.event_queries)
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.config.retrieval_dim),
            nn.Linear(self.config.retrieval_dim, self.config.retrieval_dim),
        )

    def encode_images(self, t1: Tensor, t2: Tensor) -> VisualFeatureGrid:
        return self.visual_encoder(t1, t2)

    def encode_text(self, texts: list[str], role: str = "query") -> TextFeatures:
        if self.text_encoder is None:
            raise RuntimeError("UniChangeModel has no text encoder.")
        if role not in {"query", "document"}:
            raise ValueError(f"Unsupported role: {role}")
        return self.text_encoder(texts, role=role)  # type: ignore[arg-type]

    def forward(
        self,
        t1: Tensor,
        t2: Tensor,
        texts: list[str] | None = None,
        text_role: str = "query",
        return_events: bool = True,
    ) -> UniChangeOutput:
        visual = self.encode_images(t1, t2)
        text = self.encode_text(texts, role=text_role) if texts is not None else None
        event_embeddings = event_presence = event_mask_logits = event_masks = text_conditioned_mask = None
        if return_events:
            event_embeddings, event_presence, event_mask_logits = self.event_decoder(visual.local_tokens)
            event_masks = torch.sigmoid(event_mask_logits)
            if text is not None:
                text_conditioned_mask = self.text_conditioned_mask(
                    text.global_embedding,
                    event_embeddings,
                    event_masks,
                )
        semantic_prediction = None
        if self.config.predict_semantic:
            semantic_prediction = F.normalize(self.semantic_head(visual.global_embedding), dim=-1)
        return UniChangeOutput(
            global_pair_embedding=visual.global_embedding,
            local_change_tokens=visual.local_tokens,
            text_global_embedding=None if text is None else text.global_embedding,
            text_token_embeddings=None if text is None else text.token_embeddings,
            event_embeddings=event_embeddings,
            event_presence=event_presence,
            event_mask_logits=event_mask_logits,
            event_masks=event_masks,
            text_conditioned_mask=text_conditioned_mask,
            semantic_prediction=semantic_prediction,
        )

    @staticmethod
    def text_conditioned_mask(text_embedding: Tensor, event_embeddings: Tensor, event_masks: Tensor) -> Tensor:
        if text_embedding.ndim != 2 or event_embeddings.ndim != 3 or event_masks.ndim != 3:
            raise ValueError("Expected text [N,D], events [B,K,D], masks [B,K,P].")
        if text_embedding.shape[0] == event_embeddings.shape[0]:
            weights = F.softmax((text_embedding.unsqueeze(1) * event_embeddings).sum(dim=-1), dim=-1)
            return (weights.unsqueeze(1) @ event_masks).squeeze(1)
        weights = F.softmax(torch.einsum("nd,bkd->nbk", text_embedding, event_embeddings), dim=-1)
        return torch.einsum("nbk,bkp->nbp", weights, event_masks)
