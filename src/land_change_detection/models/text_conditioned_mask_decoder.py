from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TextConditionedMaskDecoderConfig:
    hidden_dim: int = 512
    heads: int = 8
    layers: int = 2
    grid_size: int = 32
    mask_size: int = 256


class TextConditionedMaskDecoder(nn.Module):
    def __init__(self, config: TextConditionedMaskDecoderConfig | None = None):
        super().__init__()
        self.config = config or TextConditionedMaskDecoderConfig()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.layers)
        self.text_pool = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.pixel_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

    def forward(
        self,
        text_tokens: Tensor,
        change_tokens: Tensor,
        event_embeddings: Tensor,
        event_masks: Tensor,
        text_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if text_tokens.ndim != 3 or change_tokens.ndim != 3 or event_embeddings.ndim != 3:
            raise ValueError("Expected text_tokens, change_tokens and event_embeddings to be rank-3 tensors.")
        memory = torch.cat([change_tokens, event_embeddings], dim=1)
        key_padding_mask = None if text_attention_mask is None else ~text_attention_mask.to(torch.bool)
        decoded = self.decoder(text_tokens, memory, tgt_key_padding_mask=key_padding_mask)
        if text_attention_mask is None:
            pooled = decoded.mean(dim=1)
        else:
            weights = text_attention_mask.to(decoded.dtype).unsqueeze(-1)
            pooled = (decoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        query = F.normalize(self.text_pool(pooled), dim=-1)
        pixels = F.normalize(self.pixel_projection(change_tokens), dim=-1)
        low_res = torch.einsum("bd,bnd->bn", query, pixels)
        low_res = low_res.reshape(text_tokens.shape[0], self.config.grid_size, self.config.grid_size)
        event_weights = F.softmax(torch.einsum("bd,bkd->bk", query, event_embeddings), dim=-1)
        event_mask = torch.einsum("bk,bkhw->bhw", event_weights, event_masks)
        query_mask = F.interpolate(
            low_res.unsqueeze(1),
            size=event_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        fused = query_mask + event_mask
        return F.interpolate(
            fused.unsqueeze(1),
            size=(self.config.mask_size, self.config.mask_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
