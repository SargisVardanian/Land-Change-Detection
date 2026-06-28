from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DirectionalReadoutOutput:
    change_tokens: Tensor
    temporal_attention: Tensor


class DirectionalChangeReadout(nn.Module):
    """Temporal adapter for states immediately before temporal collapse."""

    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.before_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.after_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.change_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, temporal_tokens: Tensor) -> DirectionalReadoutOutput:
        if temporal_tokens.ndim != 4:
            raise ValueError("temporal_tokens must have shape [B, 2, N, D].")
        batch_size, time_steps, num_tokens, dim = temporal_tokens.shape
        if time_steps != 2:
            raise ValueError(f"DirectionalChangeReadout expects exactly before/after tokens, got T={time_steps}.")
        roles = torch.stack([self.before_role, self.after_role], dim=0)
        tokens = temporal_tokens + roles.view(1, 2, 1, dim)
        memory = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, 2, dim)
        query = self.change_query.view(1, 1, dim).expand(batch_size * num_tokens, 1, dim)
        attended, weights = self.attention(query, memory, memory, need_weights=True, average_attn_weights=True)
        change_tokens = self.ffn(attended.squeeze(1)).view(batch_size, num_tokens, dim)
        temporal_attention = weights.view(batch_size, num_tokens, 2)
        return DirectionalReadoutOutput(change_tokens=change_tokens, temporal_attention=temporal_attention)
