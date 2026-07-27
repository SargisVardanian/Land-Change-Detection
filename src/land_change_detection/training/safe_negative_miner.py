from __future__ import annotations

import torch
from torch import Tensor


def pair_caption_centroids(text_embeddings: Tensor, caption_to_pair: Tensor, num_pairs: int) -> Tensor:
    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must have shape [N,D].")
    centroids = torch.zeros(num_pairs, text_embeddings.shape[-1], device=text_embeddings.device, dtype=text_embeddings.dtype)
    counts = torch.zeros(num_pairs, device=text_embeddings.device, dtype=text_embeddings.dtype)
    centroids.index_add_(0, caption_to_pair.long(), text_embeddings)
    counts.index_add_(0, caption_to_pair.long(), torch.ones_like(caption_to_pair, dtype=text_embeddings.dtype))
    centroids = centroids / counts.clamp_min(1).unsqueeze(-1)
    return torch.nn.functional.normalize(centroids, dim=-1)


def mine_safe_negative_mask(
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    num_pairs: int,
    bottom_quantile: float = 0.35,
) -> Tensor:
    if not 0.0 < bottom_quantile < 1.0:
        raise ValueError("bottom_quantile must be in (0,1).")
    centroids = pair_caption_centroids(text_embeddings.detach(), caption_to_pair, num_pairs)
    similarities = text_embeddings.detach() @ centroids.transpose(0, 1)
    positive = torch.nn.functional.one_hot(caption_to_pair.long(), num_classes=num_pairs).bool()
    cross = ~positive
    safe_negative = torch.zeros_like(positive)
    for row_index in range(similarities.shape[0]):
        cross_scores = similarities[row_index][cross[row_index]]
        if cross_scores.numel() == 0:
            continue
        threshold = torch.quantile(cross_scores.float(), bottom_quantile)
        safe_negative[row_index] = cross[row_index] & (similarities[row_index] <= threshold.to(similarities.dtype))
    return safe_negative
