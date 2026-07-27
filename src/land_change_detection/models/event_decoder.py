from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EventDecoderConfig:
    hidden_dim: int = 512
    event_queries: int = 16
    decoder_layers: int = 2
    heads: int = 8
    ffn_dim: int = 2048
    grid_size: int = 32
    mask_size: int = 256


@dataclass(frozen=True)
class EventDecoderOutput:
    event_embeddings: Tensor
    event_presence_logits: Tensor
    event_mask_logits: Tensor


class EventDecoder(nn.Module):
    def __init__(self, config: EventDecoderConfig | None = None):
        super().__init__()
        self.config = config or EventDecoderConfig()
        self.event_queries = nn.Parameter(torch.randn(self.config.event_queries, self.config.hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.config.decoder_layers)
        self.presence_head = nn.Linear(self.config.hidden_dim, 1)
        self.mask_query_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.mask_pixel_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.mask_refine = nn.Sequential(
            nn.Conv2d(self.config.event_queries, self.config.event_queries, kernel_size=3, padding=1, groups=self.config.event_queries),
            nn.GELU(),
            nn.Conv2d(self.config.event_queries, self.config.event_queries, kernel_size=1),
        )

    def forward(self, change_tokens: Tensor, dense_skip: Tensor | None = None) -> EventDecoderOutput:
        if change_tokens.ndim != 3:
            raise ValueError(f"change_tokens must have shape [B,N,D], got {tuple(change_tokens.shape)}")
        batch_size, spatial_tokens, hidden_dim = change_tokens.shape
        if spatial_tokens != self.config.grid_size * self.config.grid_size:
            raise ValueError(f"Expected {self.config.grid_size * self.config.grid_size} tokens, got {spatial_tokens}.")
        queries = self.event_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, change_tokens)
        event_embeddings = F.normalize(decoded, dim=-1)
        event_presence_logits = self.presence_head(decoded).squeeze(-1)
        mask_queries = self.mask_query_projection(decoded)
        mask_pixels = self.mask_pixel_projection(change_tokens)
        mask_logits = torch.einsum("bkd,bnd->bkn", mask_queries, mask_pixels) / (hidden_dim**0.5)
        mask_logits = mask_logits.reshape(batch_size, self.config.event_queries, self.config.grid_size, self.config.grid_size)
        if dense_skip is not None:
            if dense_skip.ndim == 3:
                dense_skip = dense_skip.transpose(1, 2).reshape(batch_size, hidden_dim, self.config.grid_size, self.config.grid_size)
            skip = dense_skip.mean(dim=1, keepdim=True)
            mask_logits = mask_logits + skip
        mask_logits = F.interpolate(mask_logits, size=(64, 64), mode="bilinear", align_corners=False)
        mask_logits = self.mask_refine(mask_logits)
        mask_logits = F.interpolate(mask_logits, size=(self.config.mask_size, self.config.mask_size), mode="bilinear", align_corners=False)
        return EventDecoderOutput(
            event_embeddings=event_embeddings,
            event_presence_logits=event_presence_logits,
            event_mask_logits=mask_logits,
        )
