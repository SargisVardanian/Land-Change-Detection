from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_contrastive_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    row_positive_counts = positive_mask.sum(dim=1)
    valid_rows = row_positive_counts > 0
    if not torch.any(valid_rows):
        return logits.sum() * 0.0
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = log_probs.masked_fill(~positive_mask, 0.0).sum(dim=1)
    per_row = -(positive_log_probs / row_positive_counts.clamp_min(1))
    return per_row[valid_rows].mean()


def symmetric_infonce_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = image_embeddings @ text_embeddings.transpose(0, 1) / temperature
    if positive_mask is None:
        targets = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))
    if positive_mask.shape != logits.shape:
        raise ValueError(f"positive_mask shape {positive_mask.shape} does not match logits shape {logits.shape}")
    return 0.5 * (
        _masked_contrastive_loss(logits, positive_mask)
        + _masked_contrastive_loss(logits.transpose(0, 1), positive_mask.transpose(0, 1))
    )


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: list[str] | torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        label_list = [str(item) for item in labels.detach().cpu().tolist()]
    else:
        label_list = [str(item) for item in labels]
    similarities = embeddings @ embeddings.transpose(0, 1) / temperature
    mask = torch.eye(similarities.shape[0], device=similarities.device, dtype=torch.bool)
    similarities = similarities.masked_fill(mask, float("-inf"))
    losses = []
    for index, label in enumerate(label_list):
        positive_mask = torch.tensor([candidate == label for candidate in label_list], device=embeddings.device, dtype=torch.bool)
        positive_mask[index] = False
        if not torch.any(positive_mask):
            continue
        log_prob = similarities[index] - torch.logsumexp(similarities[index], dim=0)
        losses.append(-log_prob[positive_mask].mean())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def soft_histogram_contrastive_loss(
    embeddings: torch.Tensor,
    transition_histograms: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    similarities = embeddings @ embeddings.transpose(0, 1) / temperature
    target_sim = transition_histograms @ transition_histograms.transpose(0, 1)
    target_sim = target_sim / target_sim.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probs = F.log_softmax(similarities, dim=1)
    return -(target_sim * log_probs).sum(dim=1).mean()
