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


def smooth_late_interaction_score(
    text_tokens: Tensor,
    local_tokens: Tensor,
    mask: Tensor | None = None,
    tau: float = 0.05,
) -> Tensor:
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


def smooth_topk_late_interaction_score(
    text_tokens: Tensor,
    local_tokens: Tensor,
    top_k: int = 8,
    tau: float = 0.05,
    token_mask: Tensor | None = None,
) -> Tensor:
    if text_tokens.ndim != 3 or local_tokens.ndim != 3:
        raise ValueError("text_tokens and local_tokens must have shape [B, L/N, C].")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    similarity = text_tokens @ local_tokens.transpose(1, 2)
    k = min(top_k, similarity.shape[-1])
    top_values = similarity.topk(k=k, dim=-1).values
    scores = tau * torch.logsumexp(top_values / tau, dim=-1) - tau * torch.log(torch.tensor(float(k), device=similarity.device))
    if token_mask is None:
        return scores.mean(dim=-1)
    if token_mask.shape != scores.shape:
        raise ValueError("token_mask must have shape [B, L].")
    denom = token_mask.sum(dim=-1).clamp_min(1)
    return (scores * token_mask.to(scores.dtype)).sum(dim=-1) / denom


def cosine_semantic_regression(prediction: Tensor, target: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(F.normalize(prediction, dim=-1), F.normalize(target.detach(), dim=-1), dim=-1).mean()


def pair_embedding_distillation_loss(current: Tensor, previous: Tensor) -> Tensor:
    if current.shape != previous.shape:
        raise ValueError(f"current and previous must have same shape, got {current.shape} and {previous.shape}.")
    return 1.0 - F.cosine_similarity(F.normalize(current, dim=-1), F.normalize(previous.detach(), dim=-1), dim=-1).mean()


def dice_loss(mask_logits: Tensor, target_mask: Tensor, eps: float = 1e-6) -> Tensor:
    if mask_logits.shape != target_mask.shape:
        raise ValueError(f"mask_logits and target_mask must match, got {mask_logits.shape} and {target_mask.shape}.")
    probs = torch.sigmoid(mask_logits)
    target = target_mask.to(probs.dtype)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def soft_iou(pred_mask: Tensor, target_mask: Tensor, eps: float = 1e-6) -> Tensor:
    if pred_mask.shape != target_mask.shape:
        raise ValueError(f"pred_mask and target_mask must match, got {pred_mask.shape} and {target_mask.shape}.")
    pred = pred_mask.to(torch.float32)
    target = target_mask.to(torch.float32)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    return (intersection + eps) / (union + eps)


def event_overlap_loss(event_masks: Tensor) -> Tensor:
    if event_masks.ndim != 3:
        raise ValueError("event_masks must have shape [B, K, N].")
    if event_masks.shape[1] < 2:
        return event_masks.sum() * 0.0
    masks = event_masks.flatten(start_dim=2).to(torch.float32)
    masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    overlap = masks @ masks.transpose(1, 2)
    eye = torch.eye(overlap.shape[1], device=overlap.device, dtype=torch.bool).unsqueeze(0)
    return overlap.masked_fill(eye, 0.0).sum() / (overlap.shape[0] * overlap.shape[1] * (overlap.shape[1] - 1))


def event_component_coverage_loss(event_masks: Tensor, component_masks: Tensor) -> Tensor:
    if event_masks.ndim != 3 or component_masks.ndim != 3:
        raise ValueError("event_masks and component_masks must have shape [B, K/R, N].")
    if event_masks.shape[0] != component_masks.shape[0] or event_masks.shape[-1] != component_masks.shape[-1]:
        raise ValueError("event_masks and component_masks must share batch size and flattened mask size.")
    if component_masks.shape[1] == 0:
        return event_masks.sum() * 0.0
    event = event_masks.unsqueeze(2)
    comp = component_masks.to(event_masks.dtype).unsqueeze(1)
    intersection = (event * comp).sum(dim=-1)
    union = event.sum(dim=-1) + comp.sum(dim=-1) - intersection
    best_iou = ((intersection + 1e-6) / (union + 1e-6)).max(dim=1).values
    valid = component_masks.sum(dim=-1) > 0
    if not torch.any(valid):
        return event_masks.sum() * 0.0
    return (1.0 - best_iou[valid]).mean()


def variance_regularization(embeddings: Tensor, eps: float = 1e-4) -> Tensor:
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_regularization(embeddings: Tensor) -> Tensor:
    embeddings = embeddings - embeddings.mean(dim=0)
    denom = max(embeddings.shape[0] - 1, 1)
    cov = embeddings.T @ embeddings / denom
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / embeddings.shape[-1]
