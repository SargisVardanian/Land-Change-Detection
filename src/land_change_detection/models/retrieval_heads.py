from __future__ import annotations

import hashlib
import unicodedata
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


def normalize_caption_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(normalized.split())


def stable_caption_group_ids(captions: list[str], device: torch.device | str | None = None) -> Tensor:
    values: list[int] = []
    for caption in captions:
        normalized = normalize_caption_text(caption)
        digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
        values.append(int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFF_FFFF_FFFF_FFFF)
    return torch.tensor(values, dtype=torch.long, device=device)


def build_caption_positive_mask(
    caption_to_pair: Tensor,
    pair_count: int,
    caption_group_ids: Tensor | None = None,
) -> Tensor:
    mapping = caption_to_pair.long()
    if mapping.ndim != 1:
        raise ValueError(f"caption_to_pair must be one-dimensional, got {tuple(mapping.shape)}")
    if mapping.numel() == 0:
        raise ValueError("At least one caption is required for retrieval loss.")
    if int(mapping.min().item()) < 0 or int(mapping.max().item()) >= pair_count:
        raise ValueError("caption_to_pair contains an out-of-range pair index.")

    caption_count = mapping.shape[0]
    positives = torch.zeros(pair_count, caption_count, dtype=torch.bool, device=mapping.device)
    columns = torch.arange(caption_count, device=mapping.device)
    positives[mapping, columns] = True

    if caption_group_ids is None:
        return positives

    groups = caption_group_ids.to(device=mapping.device, dtype=torch.long)
    if groups.shape != mapping.shape:
        raise ValueError(
            f"caption_group_ids must have shape {tuple(mapping.shape)}, got {tuple(groups.shape)}"
        )

    _, inverse = torch.unique(groups, sorted=False, return_inverse=True)
    for group_index in range(int(inverse.max().item()) + 1):
        group_columns = torch.nonzero(inverse == group_index, as_tuple=False).flatten()
        relevant_pairs = torch.unique(mapping[group_columns])
        positives[relevant_pairs[:, None], group_columns[None, :]] = True
    return positives


def multi_positive_symmetric_info_nce(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    caption_group_ids: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    pair = F.normalize(pair_embeddings, dim=-1)
    text = F.normalize(text_embeddings, dim=-1)
    logits = pair @ text.T / temperature
    positives = build_caption_positive_mask(
        caption_to_pair,
        pair_count=pair.shape[0],
        caption_group_ids=caption_group_ids,
    )
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
