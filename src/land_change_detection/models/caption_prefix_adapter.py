from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CaptionPrefixAdapterConfig:
    hidden_dim: int = 512
    decoder_dim: int = 1024
    prefix_tokens: int = 16


class CaptionPrefixAdapter(nn.Module):
    def __init__(self, config: CaptionPrefixAdapterConfig | None = None):
        super().__init__()
        self.config = config or CaptionPrefixAdapterConfig()
        self.prefix_queries = nn.Parameter(torch.randn(self.config.prefix_tokens, self.config.hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(self.config.hidden_dim, num_heads=8, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.decoder_dim),
        )

    def forward(self, global_tokens: Tensor, event_embeddings: Tensor, change_tokens: Tensor) -> Tensor:
        if global_tokens.ndim != 3 or event_embeddings.ndim != 3 or change_tokens.ndim != 3:
            raise ValueError("Expected global_tokens, event_embeddings and change_tokens to have shape [B,N,D].")
        batch_size = global_tokens.shape[0]
        memory = torch.cat([global_tokens, event_embeddings, change_tokens.mean(dim=1, keepdim=True)], dim=1)
        queries = self.prefix_queries.unsqueeze(0).expand(batch_size, -1, -1)
        prefix, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        return self.projection(prefix)
