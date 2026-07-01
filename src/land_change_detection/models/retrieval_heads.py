from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalHeadOutput:
    pair_embedding: Tensor
    text_embedding: Tensor | None
    logits: Tensor | None


class RetrievalProjectionHead(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.pair_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, pair_embedding: Tensor, text_embedding: Tensor | None = None) -> RetrievalHeadOutput:
        pair = F.normalize(self.pair_projection(pair_embedding), dim=-1)
        text = F.normalize(text_embedding, dim=-1) if text_embedding is not None else None
        logits = pair @ text.T if text is not None else None
        return RetrievalHeadOutput(pair_embedding=pair, text_embedding=text, logits=logits)


def multi_positive_symmetric_info_nce(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    pair = F.normalize(pair_embeddings, dim=-1)
    text = F.normalize(text_embeddings, dim=-1)
    logits = pair @ text.T / temperature
    positives = torch.zeros_like(logits, dtype=torch.bool)
    positives[caption_to_pair.long(), torch.arange(text.shape[0], device=text.device)] = True
    pair_log_prob = logits.log_softmax(dim=1)
    text_log_prob = logits.T.log_softmax(dim=1)
    pair_loss = -(pair_log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1)).mean()
    text_positives = positives.T
    text_loss = -(text_log_prob.masked_fill(~text_positives, 0.0).sum(dim=1) / text_positives.sum(dim=1).clamp_min(1)).mean()
    return 0.5 * (pair_loss + text_loss)


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positives = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits.log_softmax(dim=1)
    valid = positives.sum(dim=1) > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    losses = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid] / positives.sum(dim=1)[valid])
    return losses.mean()
