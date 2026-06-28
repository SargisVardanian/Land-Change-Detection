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
    text_attention_mask: Tensor | None
    event_embeddings: Tensor | None
    event_presence: Tensor | None
    event_mask_logits: Tensor | None
    event_masks: Tensor | None
    text_conditioned_mask: Tensor | None
    semantic_prediction: Tensor | None
    direction_logits: Tensor | None

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
    mask_dim: int = 256
    predict_semantic: bool = True


class ChangeEventDecoder(nn.Module):
    def __init__(self, retrieval_dim: int = 512, event_queries: int = 32, mask_dim: int = 256):
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
        self.mask_query_projection = nn.Linear(retrieval_dim, mask_dim)
        self.mask_pixel_projection = nn.Linear(retrieval_dim, mask_dim)
        self.mask_logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, local_tokens: Tensor, time_embedding: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = local_tokens.shape[0]
        queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        if time_embedding is not None:
            queries = queries + time_embedding.unsqueeze(1)
        decoded = self.decoder(queries, local_tokens)
        event_embeddings = F.normalize(decoded, dim=-1)
        presence = self.presence_head(decoded).squeeze(-1)
        mask_queries = self.mask_query_projection(decoded)
        mask_pixels = self.mask_pixel_projection(local_tokens)
        mask_logits = self.mask_logit_scale.exp().clamp(max=100.0) * torch.einsum("bkd,bpd->bkp", mask_queries, mask_pixels) / (mask_queries.shape[-1] ** 0.5)
        return event_embeddings, presence, mask_logits


class TemporalConditionEncoder(nn.Module):
    def __init__(self, retrieval_dim: int = 512):
        super().__init__()
        self.role_embedding = nn.Embedding(2, retrieval_dim)
        self.known_embedding = nn.Embedding(2, retrieval_dim)
        self.delta_projection = nn.Sequential(nn.Linear(1, retrieval_dim), nn.Tanh())
        self.direction_head = nn.Linear(retrieval_dim, 2)

    def forward(self, temporal_context: list[dict] | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if temporal_context is None:
            temporal_context = [{"before_index": 0, "after_index": 1, "delta_days": None, "timestamps_known": False} for _ in range(batch_size)]
        before = torch.tensor([int(item.get("before_index", 0)) for item in temporal_context], device=device).clamp(0, 1)
        after = torch.tensor([int(item.get("after_index", 1)) for item in temporal_context], device=device).clamp(0, 1)
        known = torch.tensor([bool(item.get("timestamps_known", False)) for item in temporal_context], device=device, dtype=torch.long)
        delta = torch.tensor(
            [0.0 if item.get("delta_days") is None else abs(float(item["delta_days"])) for item in temporal_context],
            device=device,
            dtype=dtype,
        ).view(batch_size, 1)
        time_embedding = (
            self.role_embedding(after).to(dtype)
            - self.role_embedding(before).to(dtype)
            + self.known_embedding(known).to(dtype)
            + self.delta_projection(torch.log1p(delta))
        )
        return time_embedding, self.direction_head(time_embedding)


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
        self.event_decoder = ChangeEventDecoder(self.config.retrieval_dim, self.config.event_queries, self.config.mask_dim)
        self.time_encoder = TemporalConditionEncoder(self.config.retrieval_dim)
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.config.retrieval_dim),
            nn.Linear(self.config.retrieval_dim, self.config.retrieval_dim),
        )

    def train(self, mode: bool = True) -> UniChangeModel:
        super().train(mode)
        self._keep_frozen_backbones_eval()
        return self

    def _keep_frozen_backbones_eval(self) -> None:
        self.visual_encoder.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
            base = getattr(self.text_encoder, "model", None)
            if base is not None:
                base.eval()

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
        temporal_context: list[dict] | None = None,
        return_events: bool = True,
    ) -> UniChangeOutput:
        visual = self.encode_images(t1, t2)
        text = self.encode_text(texts, role=text_role) if texts is not None else None
        time_embedding, direction_logits = self.time_encoder(
            temporal_context,
            batch_size=visual.global_embedding.shape[0],
            device=visual.global_embedding.device,
            dtype=visual.global_embedding.dtype,
        )
        global_pair_embedding = F.normalize(visual.global_embedding + time_embedding, dim=-1)
        event_embeddings = event_presence = event_mask_logits = event_masks = text_conditioned_mask = None
        if return_events:
            event_embeddings, event_presence, event_mask_logits = self.event_decoder(visual.local_tokens, time_embedding=time_embedding)
            event_masks = torch.sigmoid(event_mask_logits)
            if text is not None:
                text_conditioned_mask = self.text_conditioned_mask(
                    text.global_embedding,
                    event_embeddings,
                    event_masks,
                )
        semantic_prediction = None
        if self.config.predict_semantic:
            semantic_prediction = F.normalize(self.semantic_head(global_pair_embedding), dim=-1)
        return UniChangeOutput(
            global_pair_embedding=global_pair_embedding,
            local_change_tokens=visual.local_tokens,
            text_global_embedding=None if text is None else text.global_embedding,
            text_token_embeddings=None if text is None else text.token_embeddings,
            text_attention_mask=None if text is None else text.attention_mask,
            event_embeddings=event_embeddings,
            event_presence=event_presence,
            event_mask_logits=event_mask_logits,
            event_masks=event_masks,
            text_conditioned_mask=text_conditioned_mask,
            semantic_prediction=semantic_prediction,
            direction_logits=direction_logits,
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
