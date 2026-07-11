from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def temporal_patch_descriptor(per_time_tokens: Tensor) -> Tensor:
    """Build D_p=[V1,V2,V2-V1,abs(V2-V1),coordinates]."""
    if per_time_tokens.ndim != 4 or per_time_tokens.shape[1] != 2:
        raise ValueError("per_time_tokens must have shape [B,2,N,D]")
    before, after = per_time_tokens[:, 0], per_time_tokens[:, 1]
    side = int(math.isqrt(before.shape[1]))
    if side * side != before.shape[1]:
        raise ValueError("Temporal explanation patches must form a square grid")
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, side, device=before.device, dtype=before.dtype),
        torch.linspace(0, 1, side, device=before.device, dtype=before.dtype), indexing="ij",
    )
    coords = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).expand(before.shape[0], -1, -1)
    return torch.cat((before, after, after - before, (after - before).abs(), coords), dim=-1)


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

    @staticmethod
    def masked_local_embedding(patch_tokens: Tensor, query_mask_logits: Tensor) -> Tensor:
        """Pool candidate patches with the exact mask weights exposed to users."""
        weights = query_mask_logits.sigmoid()
        pooled = torch.einsum("qbn,bnd->qbd", weights, patch_tokens)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.normalize(pooled, dim=-1)

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
        local_embeddings = self.masked_local_embedding(patch_tokens, query_mask_logits)
        local_score = torch.einsum("qd,qbd->qb", queries, local_embeddings)
        final_score = self.alpha * global_score + self.beta * local_score
        return {
            "global_score": global_score,
            "local_score": local_score,
            "final_score": final_score,
            "query_mask_logits": query_mask_logits,
            "local_embeddings": local_embeddings,
            "mask_query_embeddings": mask_queries / self.logit_scale,
            "score_mode": "fused",
        }


def _per_query_segmentation_losses(query_mask_logits: Tensor, caption_to_pair: Tensor, pair_masks: Tensor) -> Tensor:
    """Return BCE-plus-Dice loss for every query against its paired target."""
    query_count, pair_count, patch_count = query_mask_logits.shape
    if caption_to_pair.shape != (query_count,):
        raise ValueError("caption_to_pair must contain one pair index per query")
    if pair_masks.ndim != 3 or pair_masks.shape[0] != pair_count:
        raise ValueError("pair_masks must have shape [B,H,W] with the QCPR pair count")
    side = int(math.isqrt(patch_count))
    if side * side != patch_count:
        raise ValueError("QCPR patch count must form a square grid")
    targets = F.interpolate(pair_masks[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1)
    rows = torch.arange(query_count, device=query_mask_logits.device)
    mapping = caption_to_pair.long()
    logits = query_mask_logits[rows, mapping]
    target = targets.to(logits.device)[mapping]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=1)
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.sum(dim=1) + target.sum(dim=1) + 1.0))
    return bce + dice


def segmentation_loss_components(
    query_mask_logits: Tensor,
    caption_to_pair: Tensor,
    pair_masks: Tensor,
    pair_target_kinds: list[str],
    pair_supervision_weights: Tensor,
) -> dict[str, Tensor | int | float]:
    """Compute separated query-specific and generic mask supervision losses."""
    if query_mask_logits.ndim != 3:
        raise ValueError("query_mask_logits must have shape [Q,B,N]")
    query_count, pair_count, _ = query_mask_logits.shape
    if len(pair_target_kinds) != pair_count:
        raise ValueError("pair_target_kinds must contain one kind per pair")
    if pair_supervision_weights.shape != (pair_count,):
        raise ValueError("pair_supervision_weights must contain one weight per pair")
    per_query = _per_query_segmentation_losses(query_mask_logits, caption_to_pair, pair_masks)
    mapping = caption_to_pair.long()
    weights = pair_supervision_weights.to(device=per_query.device, dtype=per_query.dtype)[mapping]
    query_specific_pairs = torch.tensor(
        [kind == "query_specific" for kind in pair_target_kinds], device=per_query.device, dtype=torch.bool
    )
    generic_pairs = torch.tensor(
        [kind in {"binary_generic", "semantic_transition_union"} for kind in pair_target_kinds],
        device=per_query.device,
        dtype=torch.bool,
    )

    def weighted(group_pairs: Tensor) -> Tensor:
        keep = group_pairs[mapping] & (weights > 0)
        if not torch.any(keep):
            return per_query.sum() * 0.0
        return (per_query[keep] * weights[keep]).mean()

    query_loss = weighted(query_specific_pairs)
    generic_loss = weighted(generic_pairs)
    pair_has_weight = pair_supervision_weights.to(query_specific_pairs.device) > 0
    query_specific_supervised = query_specific_pairs & pair_has_weight
    generic_supervised = generic_pairs & pair_has_weight
    all_pairs = query_specific_supervised | generic_supervised
    all_keep = all_pairs[mapping] & (weights > 0)
    total_loss = (
        (per_query[all_keep] * weights[all_keep]).mean()
        if torch.any(all_keep)
        else per_query.sum() * 0.0
    )
    return {
        "query_specific_segmentation_loss": query_loss,
        "generic_change_segmentation_loss": generic_loss,
        "total_segmentation_loss": total_loss,
        "segmentation_supervised_pairs": int(all_pairs.sum().item()),
        "query_specific_supervised_pairs": int(query_specific_supervised.sum().item()),
        "generic_supervised_pairs": int(generic_supervised.sum().item()),
        "mean_segmentation_weight": float(pair_supervision_weights[all_pairs].float().mean().item()) if torch.any(all_pairs) else 0.0,
    }


def query_segmentation_loss(
    query_mask_logits: Tensor,
    caption_to_pair: Tensor,
    pair_masks: Tensor,
    pair_supervision_mask: Tensor | None = None,
) -> Tensor:
    if query_mask_logits.ndim != 3:
        raise ValueError("query_mask_logits must have shape [Q,B,N]")
    query_count, pair_count, _ = query_mask_logits.shape
    per_query = _per_query_segmentation_losses(query_mask_logits, caption_to_pair, pair_masks)
    mapping = caption_to_pair.long()
    if pair_supervision_mask is not None:
        if pair_supervision_mask.shape != (pair_count,):
            raise ValueError("pair_supervision_mask must contain one flag per pair")
        keep = pair_supervision_mask.to(device=query_mask_logits.device, dtype=torch.bool)[mapping]
        if not torch.any(keep):
            return query_mask_logits.sum() * 0.0
        return per_query[keep].mean()
    return per_query.mean()
