from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import TemporalSigLIPConfig


@dataclass(frozen=True)
class SoftChangeMapOutput:
    map_logits: Tensor
    map_probabilities: Tensor
    change_tokens: Tensor
    localization_temperature: Tensor


class TemporalSoftChangeMap(nn.Module):
    """Post-retrieval diagnostic map from the same temporal patch outputs."""

    def __init__(self, config: TemporalSigLIPConfig | None = None) -> None:
        super().__init__()
        resolved = (config or TemporalSigLIPConfig()).validate()
        self.hidden_size = resolved.hidden_size
        self.patch_grid = resolved.patch_grid
        self.log_localization_temperature = nn.Parameter(
            torch.tensor(resolved.localization_temperature, dtype=torch.float32).log()
        )

    @property
    def localization_temperature(self) -> Tensor:
        return self.log_localization_temperature.clamp(-5.0, 2.0).exp()

    def forward(
        self,
        temporal_tokens: Tensor,
        text_embedding: Tensor,
    ) -> SoftChangeMapOutput:
        if temporal_tokens.ndim != 4:
            raise ValueError("temporal_tokens must have shape [B,T,N,D]")
        if text_embedding.ndim != 2:
            raise ValueError("text_embedding must have shape [B,D]")
        batch, frames, patches, hidden = temporal_tokens.shape
        if hidden != self.hidden_size or patches != self.patch_grid * self.patch_grid:
            raise ValueError("temporal tokens do not match the localization contract")
        if text_embedding.shape != (batch, hidden):
            raise ValueError("text embedding must match temporal batch and hidden size")
        if frames < 2:
            raise ValueError("a temporal change map requires at least two frames")
        # LayerNorm without affine parameters keeps the only trainable map
        # parameter as the scalar localization temperature.
        changes = F.layer_norm(
            temporal_tokens[:, 1:] - temporal_tokens[:, :-1],
            (hidden,),
        )
        text = F.normalize(text_embedding, dim=-1)
        logits = torch.einsum("bihd,bd->bih", changes, text)
        logits = logits / self.localization_temperature.to(dtype=logits.dtype)
        logits = logits.reshape(batch, frames - 1, self.patch_grid, self.patch_grid)
        return SoftChangeMapOutput(
            map_logits=logits,
            map_probabilities=torch.sigmoid(logits),
            change_tokens=changes,
            localization_temperature=self.localization_temperature,
        )
