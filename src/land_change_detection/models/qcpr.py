from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QCPRPatchReranker(nn.Module):
    """Query-conditioned patch reranker with an auxiliary mask head."""

    def __init__(self, hidden_dim: int = 512, retrieval_dim: int = 512, *, alpha: float = 1.0, beta: float = 0.25):
        super().__init__()
        if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
            raise ValueError("QCPR fusion weights must be non-negative and not both zero")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.patch_projector = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, retrieval_dim))
        self.query_mask_head = nn.Sequential(nn.LayerNorm(retrieval_dim), nn.Linear(retrieval_dim, retrieval_dim))
        self.logit_scale = 1.0 / math.sqrt(float(retrieval_dim))

    def project_patches(self, change_tokens: Tensor) -> Tensor:
        if change_tokens.ndim != 3:
            raise ValueError("change_tokens must have shape [B,N,D]")
        return F.normalize(self.patch_projector(change_tokens), dim=-1)

    def score(self, query_embeddings: Tensor, pair_embeddings: Tensor, patch_tokens: Tensor) -> dict[str, Tensor | str]:
        if query_embeddings.ndim != 2 or pair_embeddings.ndim != 2 or patch_tokens.ndim != 3:
            raise ValueError("QCPR expects queries [Q,D], pairs [B,D], and patches [B,N,D]")
        queries = F.normalize(query_embeddings, dim=-1)
        pairs = F.normalize(pair_embeddings, dim=-1)
        global_score = queries @ pairs.T
        mask_queries = F.normalize(self.query_mask_head(queries), dim=-1)
        query_mask_logits = torch.einsum("qd,bnd->qbn", mask_queries, patch_tokens) / self.logit_scale
        local_score = query_mask_logits.sigmoid().amax(dim=-1)
        final_score = self.alpha * global_score + self.beta * local_score
        return {
            "global_score": global_score,
            "local_score": local_score,
            "final_score": final_score,
            "query_mask_logits": query_mask_logits,
            "mask_query_embeddings": mask_queries / self.logit_scale,
            "score_mode": "fused",
        }


def query_segmentation_loss(
    query_mask_logits: Tensor,
    caption_to_pair: Tensor,
    pair_masks: Tensor,
    pair_supervision_mask: Tensor | None = None,
) -> Tensor:
    if query_mask_logits.ndim != 3:
        raise ValueError("query_mask_logits must have shape [Q,B,N]")
    if pair_masks.ndim != 3:
        raise ValueError("pair_masks must have shape [B,H,W]")
    query_count, pair_count, patch_count = query_mask_logits.shape
    if caption_to_pair.shape != (query_count,):
        raise ValueError("caption_to_pair must contain one pair index per query")
    if pair_masks.shape[0] != pair_count:
        raise ValueError("pair_masks and query_mask_logits must have the same pair count")
    side = int(math.isqrt(patch_count))
    if side * side != patch_count:
        raise ValueError("QCPR patch count must form a square grid")
    targets = F.interpolate(pair_masks[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1)
    rows = torch.arange(query_count, device=query_mask_logits.device)
    mapping = caption_to_pair.long()
    if pair_supervision_mask is not None:
        if pair_supervision_mask.shape != (pair_count,):
            raise ValueError("pair_supervision_mask must contain one flag per pair")
        keep = pair_supervision_mask.to(device=query_mask_logits.device, dtype=torch.bool)[mapping]
        if not torch.any(keep):
            return query_mask_logits.sum() * 0.0
        rows = rows[keep]
        mapping = mapping[keep]
    matched_logits = query_mask_logits[rows, mapping]
    matched_targets = targets.to(matched_logits.device)[mapping]
    bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
    probabilities = matched_logits.sigmoid()
    intersection = (probabilities * matched_targets).sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.sum(dim=1) + matched_targets.sum(dim=1) + 1.0))
    return bce + dice.mean()
