"""Frozen Jina v5 query encoder with an identity-preserving adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryFeatures:
    text_cls: Tensor
    text_tokens: Tensor
    text_mask: Tensor
    model_identifier: str = "configured-jina-v5"

    def validate(self) -> None:
        if self.text_tokens.ndim != 3 or self.text_cls.ndim != 2:
            raise ValueError("query tensors must be [B,L,D] and [B,D]")
        if self.text_mask.shape != self.text_tokens.shape[:2] or self.text_mask.dtype != torch.bool:
            raise ValueError("text_mask must be boolean [B,L]")
        if self.text_cls.shape != (self.text_tokens.shape[0], self.text_tokens.shape[2]):
            raise ValueError("text_cls shape does not match text_tokens")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class JinaQueryEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        adapter_layers: int = 2,
        heads: int = 8,
        ff_ratio: int = 4,
        dropout: float = 0.1,
        base_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.base_encoder = base_encoder
        self.input_projection = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * ff_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.adapter = nn.TransformerEncoder(layer, num_layers=adapter_layers)
        self.adapter_residual_scale = nn.Parameter(torch.zeros(()))
        self.adapter_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if self.base_encoder is not None:
            self.base_encoder.eval()
            for parameter in self.base_encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_encoder is not None:
            self.base_encoder.eval()
        return self

    def _read_base(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        if self.base_encoder is None:
            raise ValueError("text strings require a configured base_encoder")
        output: Any = self.base_encoder(texts)
        if isinstance(output, dict):
            return output["text_tokens"], output["text_mask"]
        if isinstance(output, (tuple, list)) and len(output) == 2:
            return output[0], output[1]
        raise TypeError("base_encoder must return a mapping or (tokens, mask)")

    def forward(self, tokens: Tensor | list[str], mask: Tensor | None = None) -> QueryFeatures:
        if isinstance(tokens, (list, tuple)):
            token_tensor, mask_tensor = self._read_base(list(tokens))
        else:
            token_tensor = tokens
            mask_tensor = mask
        if not isinstance(token_tensor, Tensor):
            raise TypeError("tokens must be a tensor or a list of strings")
        if mask_tensor is None:
            raise ValueError("text mask is required")
        if token_tensor.ndim != 3 or mask_tensor.shape != token_tensor.shape[:2]:
            raise ValueError("tokens must be [B,L,D] and mask [B,L]")
        mask_tensor = mask_tensor.to(dtype=torch.bool, device=token_tensor.device)
        x = self.input_projection(token_tensor)
        delta = self.adapter(self.adapter_norm(x), src_key_padding_mask=~mask_tensor)
        x = x + self.adapter_residual_scale * delta
        cls = F.normalize(self.output_projection(masked_mean(x, mask_tensor)), dim=-1)
        result = QueryFeatures(cls, x, mask_tensor)
        result.validate()
        return result
