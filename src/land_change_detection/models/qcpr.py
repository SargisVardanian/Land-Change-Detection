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

    def __init__(self, hidden_dim: int = 512, retrieval_dim: int = 512, *, alpha: float = 1.0, beta: float = 0.25, architecture_version: str = "v1"):
        super().__init__()
        if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
            raise ValueError("QCPR fusion weights must be non-negative and not both zero")
        if architecture_version not in {"v1", "v2"}:
            raise ValueError("qcpr architecture_version must be 'v1' or 'v2'")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.architecture_version = architecture_version
        self.patch_projector = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, retrieval_dim))
        self.query_mask_head = nn.Sequential(nn.LayerNorm(retrieval_dim), nn.Linear(retrieval_dim, retrieval_dim))
        self.logit_scale = 1.0 / math.sqrt(float(retrieval_dim))
        if architecture_version == "v2":
            self.temporal_descriptor_mlp = nn.Sequential(nn.LayerNorm(4 * hidden_dim + 2), nn.Linear(4 * hidden_dim + 2, retrieval_dim), nn.GELU(), nn.Linear(retrieval_dim, retrieval_dim))
            self.token_projection = nn.Linear(retrieval_dim, retrieval_dim)
            self.interaction_mlp = nn.Sequential(nn.LayerNorm(3 * retrieval_dim), nn.Linear(3 * retrieval_dim, retrieval_dim), nn.GELU(), nn.Linear(retrieval_dim, 1))
            self.temporal_channel_head = nn.Sequential(nn.LayerNorm(retrieval_dim), nn.Linear(retrieval_dim, retrieval_dim), nn.GELU(), nn.Linear(retrieval_dim, 3))

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

    def score_v2(self, query_embeddings: Tensor, query_token_embeddings: Tensor, query_attention_mask: Tensor, pair_embeddings: Tensor, per_time_tokens: Tensor) -> dict[str, Tensor | str]:
        if self.architecture_version != "v2":
            raise RuntimeError("score_v2 requires qcpr architecture v2")
        descriptors = F.normalize(self.temporal_descriptor_mlp(temporal_patch_descriptor(per_time_tokens)), dim=-1)
        token_embeddings = F.normalize(self.token_projection(query_token_embeddings), dim=-1)
        affinity = torch.einsum("bnd,qld->qbnl", descriptors, token_embeddings) / self.logit_scale
        affinity = affinity.masked_fill(~query_attention_mask[:, None, None, :].bool(), -1e4)
        attended_query = torch.einsum("qbnl,qld->qbnd", affinity.softmax(dim=-1), token_embeddings)
        descriptor = descriptors.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1, -1)
        query_mask_logits = self.interaction_mlp(torch.cat((descriptor, attended_query, descriptor * attended_query), dim=-1)).squeeze(-1)
        local_embeddings = self.masked_local_embedding(descriptors, query_mask_logits)
        queries = F.normalize(query_embeddings, dim=-1)
        global_score = queries @ F.normalize(pair_embeddings, dim=-1).T
        local_score = torch.einsum("qd,qbd->qb", queries, local_embeddings)
        return {
            "global_score": global_score,
            "local_score": local_score,
            "token_patch_score": query_mask_logits.sigmoid().amax(dim=-1),
            "final_score": self.alpha * global_score + self.beta * local_score,
            "query_mask_logits": query_mask_logits,
            "patch_tokens": descriptors,
            "temporal_explanation_logits": self.temporal_channel_head(descriptors),
            "score_mode": "qcpr_v2",
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


def temporal_channel_loss_components(
    logits: Tensor,
    pair_masks: Tensor,
    changed_masks: Tensor,
    pair_target_kinds: list[str],
    pair_supervision_weights: Tensor,
    change_types: list[str | None],
    reverse_logits: Tensor | None = None,
) -> dict[str, Tensor | int]:
    """Supervise changed/appeared/disappeared channels with real labels only."""
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("temporal logits must have shape [B,N,3]")
    batch, patches, _ = logits.shape
    side = int(math.isqrt(patches))
    if side * side != patches or pair_masks.shape[0] != batch:
        raise ValueError("temporal channel targets do not align")
    targets = F.interpolate(pair_masks[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1).to(logits.device)
    changed_targets = F.interpolate(changed_masks[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1).to(logits.device)
    base_weights = pair_supervision_weights.to(logits.device, logits.dtype)

    def channel_loss(channel: int, keep: Tensor, weights: Tensor, target_values: Tensor | None = None) -> Tensor:
        keep = keep & (weights > 0)
        if target_values is None:
            target_values = targets
        if not torch.any(keep):
            return logits.sum() * 0.0
        prediction = logits[keep, :, channel]
        target = target_values[keep]
        bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none").mean(dim=1)
        probability = prediction.sigmoid()
        dice = 1.0 - (2 * (probability * target).sum(1) + 1) / (probability.sum(1) + target.sum(1) + 1)
        return ((bce + dice) * weights[keep]).mean()

    real = torch.tensor([kind != "none" for kind in pair_target_kinds], device=logits.device)
    appeared = torch.tensor([value == "appeared" for value in change_types], device=logits.device)
    disappeared = torch.tensor([value == "disappeared" for value in change_types], device=logits.device)
    changed_loss = channel_loss(0, real, base_weights, changed_targets)
    appeared_loss = channel_loss(1, appeared, torch.where(appeared, torch.ones_like(base_weights), torch.zeros_like(base_weights)))
    disappeared_loss = channel_loss(2, disappeared, torch.where(disappeared, torch.ones_like(base_weights), torch.zeros_like(base_weights)))
    reversal = logits.sum() * 0.0
    if reverse_logits is not None:
        supervised = real & (base_weights > 0)
        if torch.any(supervised):
            reversal = F.mse_loss(logits[supervised, :, 0].sigmoid(), reverse_logits[supervised, :, 0].sigmoid())
            reversal = reversal + F.mse_loss(logits[supervised, :, 1].sigmoid(), reverse_logits[supervised, :, 2].sigmoid())
            reversal = reversal + F.mse_loss(logits[supervised, :, 2].sigmoid(), reverse_logits[supervised, :, 1].sigmoid())
    return {"changed_channel_loss": changed_loss, "appeared_channel_loss": appeared_loss, "disappeared_channel_loss": disappeared_loss, "temporal_reversal_consistency_loss": reversal, "temporal_supervised_pairs": int(real.sum().item())}
