from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_multi_positive_sigmoid_loss(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    safe_negative_mask: Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[Tensor, dict[str, float]]:
    if pair_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("pair_embeddings and text_embeddings must be rank-2 tensors.")
    if caption_to_pair.ndim != 1 or caption_to_pair.shape[0] != text_embeddings.shape[0]:
        raise ValueError("caption_to_pair must be rank-1 with one entry per caption.")
    logits = text_embeddings @ pair_embeddings.transpose(0, 1) / temperature
    positive_mask = F.one_hot(caption_to_pair.long(), num_classes=pair_embeddings.shape[0]).bool()
    if safe_negative_mask is None:
        negative_mask = torch.zeros_like(positive_mask)
    else:
        if safe_negative_mask.shape != positive_mask.shape:
            raise ValueError("safe_negative_mask must have shape [num_captions, num_pairs].")
        negative_mask = safe_negative_mask.bool() & ~positive_mask
    known_mask = positive_mask | negative_mask
    targets = positive_mask.to(logits.dtype)
    if not torch.any(known_mask):
        return logits.sum() * 0.0, {"positive_ratio": 0.0, "negative_ratio": 0.0, "ignored_ratio": 1.0}
    loss = F.binary_cross_entropy_with_logits(logits[known_mask], targets[known_mask])
    total = known_mask.numel()
    return loss, {
        "positive_ratio": float(positive_mask.sum().item() / total),
        "negative_ratio": float(negative_mask.sum().item() / total),
        "ignored_ratio": float((~known_mask).sum().item() / total),
    }


def smooth_late_interaction_score(text_tokens: Tensor, local_tokens: Tensor, mask: Tensor | None = None, tau: float = 0.05) -> Tensor:
    if text_tokens.ndim != 3 or local_tokens.ndim != 3:
        raise ValueError("text_tokens and local_tokens must have shape [B, L/N, C].")
    similarity = text_tokens @ local_tokens.transpose(1, 2)
    if mask is not None:
        similarity = similarity.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
    token_scores = tau * torch.logsumexp(similarity / tau, dim=-1)
    if mask is None:
        return token_scores.mean(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (token_scores * mask.to(token_scores.dtype)).sum(dim=-1) / denom


def cosine_semantic_regression(prediction: Tensor, target: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(F.normalize(prediction, dim=-1), F.normalize(target.detach(), dim=-1), dim=-1).mean()


def variance_regularization(embeddings: Tensor, eps: float = 1e-4) -> Tensor:
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_regularization(embeddings: Tensor) -> Tensor:
    embeddings = embeddings - embeddings.mean(dim=0)
    denom = max(embeddings.shape[0] - 1, 1)
    cov = embeddings.T @ embeddings / denom
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / embeddings.shape[-1]
