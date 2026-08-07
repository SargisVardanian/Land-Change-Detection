from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import TemporalSigLIPBackbone
from .config import TemporalSigLIPConfig
from .temporal_pair import TemporalPairEncoder, TemporalPairOutput


@dataclass(frozen=True)
class TemporalSigLIPOutput:
    score_matrix: Tensor
    pair_embedding: Tensor
    text_embedding: Tensor
    temporal_tokens: Tensor
    text_tokens: Tensor
    text_mask: Tensor


class TemporalSigLIP(nn.Module):
    """The active direct retrieval model.

    The primary score is exactly a scaled cosine similarity between one
    query-independent pair embedding and one text embedding.  There is no
    evidence branch, relevance MLP, late interaction, or mandatory reranker.
    """

    def __init__(
        self,
        backbone: TemporalSigLIPBackbone | None = None,
        config: TemporalSigLIPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = (config or TemporalSigLIPConfig()).validate()
        self.backbone = backbone
        self.temporal_pair = TemporalPairEncoder(self.config)
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.initial_logit_scale, dtype=torch.float32)
        )
        if backbone is not None and backbone.hidden_size != self.config.hidden_size:
            raise ValueError("backbone hidden size does not match TemporalSigLIP config")

    @property
    def effective_logit_scale(self) -> Tensor:
        return self.logit_scale.clamp(
            min=self.config.min_logit_scale,
            max=self.config.max_logit_scale,
        ).exp()

    def encode_pair_from_features(
        self,
        frame_tokens: Tensor,
        *,
        timestamps: Tensor | None = None,
    ) -> TemporalPairOutput:
        return self.temporal_pair(frame_tokens, timestamps=timestamps)

    def encode_query_from_features(
        self,
        text_tokens: Tensor,
        text_embedding: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if text_tokens.ndim != 3 or text_embedding.ndim != 2 or text_mask.ndim != 2:
            raise ValueError("text features must be [Q,L,D], [Q,D] and [Q,L]")
        if text_tokens.shape[:2] != text_mask.shape:
            raise ValueError("text_mask must match text_tokens")
        if text_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError("text token hidden size does not match config")
        return (
            F.normalize(text_embedding, dim=-1),
            text_tokens,
            text_mask.bool(),
        )

    def score_embeddings(self, text_embedding: Tensor, pair_embedding: Tensor) -> Tensor:
        text = F.normalize(text_embedding, dim=-1)
        pair = F.normalize(pair_embedding, dim=-1)
        if text.ndim != 2 or pair.ndim != 2 or text.shape[-1] != pair.shape[-1]:
            raise ValueError("embeddings must be [Q,D] and [P,D]")
        return self.effective_logit_scale * (text @ pair.transpose(0, 1))

    def forward_from_features(
        self,
        frame_tokens: Tensor,
        text_tokens: Tensor,
        text_embeddings: Tensor,
        text_mask: Tensor,
        *,
        timestamps: Tensor | None = None,
    ) -> TemporalSigLIPOutput:
        pair = self.encode_pair_from_features(frame_tokens, timestamps=timestamps)
        text_embedding, tokens, mask = self.encode_query_from_features(
            text_tokens, text_embeddings, text_mask
        )
        scores = self.score_embeddings(text_embedding, pair.pair_embedding)
        return TemporalSigLIPOutput(
            score_matrix=scores,
            pair_embedding=pair.pair_embedding,
            text_embedding=text_embedding,
            temporal_tokens=pair.temporal_tokens,
            text_tokens=tokens,
            text_mask=mask,
        )

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        pixel_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> TemporalSigLIPOutput:
        if self.backbone is None:
            raise RuntimeError("raw-input forward requires a TemporalSigLIPBackbone")
        image = self.backbone.encode_images(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        text = self.backbone.encode_text(input_ids, attention_mask)
        return self.forward_from_features(
            image.patch_tokens,
            text.token_embeddings,
            text.pooled_embedding,
            text.attention_mask,
            timestamps=timestamps,
        )

    def set_stage_a(self) -> None:
        if self.backbone is not None:
            self.backbone.freeze_towers()

    def set_stage_b(self) -> None:
        if self.backbone is None:
            raise RuntimeError("Stage B requires an attached pretrained backbone")
        self.backbone.enable_stage_b(top_blocks=2)

    def trainable_parameter_report(self) -> dict[str, dict[str, int]]:
        names = {
            "temporal_pair": self.temporal_pair,
            "logit_scale": nn.ParameterList([self.logit_scale]),
        }
        report: dict[str, dict[str, int]] = {}
        for name, module in names.items():
            parameters = list(module.parameters())
            report[name] = {
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "trainable_count": sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
            }
        if self.backbone is not None:
            report["siglip2_towers"] = self.backbone.trainable_scope()
        return report
