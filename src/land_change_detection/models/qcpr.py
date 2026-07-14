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


def _structured_signature(caption: str) -> dict[str, object]:
    from land_change_detection.models.retrieval_heads import classify_caption_semantics

    semantics = classify_caption_semantics(caption)
    if semantics["no_change"]:
        direction = "no_change"
    elif semantics["appeared"] or semantics["constructed"] or semantics["added"]:
        direction = "appeared"
    elif semantics["disappeared"] or semantics["demolished"] or semantics["removed"]:
        direction = "disappeared"
    elif semantics["increased"] or semantics["expanded"]:
        direction = "increased"
    elif semantics["decreased"] or semantics["reduced"]:
        direction = "decreased"
    else:
        direction = "changed" if semantics["changed"] else "unknown"
    normalized = caption.casefold()
    relation = "replacement" if any(term in normalized for term in ("replace", "in place of", "converted to")) else "none"
    aliases = {
        "buildings": "building", "houses": "house", "roads": "road", "fields": "field",
        "crops": "crop", "trees": "tree", "greenhouses": "greenhouse", "structures": "structure",
    }
    return {
        "objects": frozenset(aliases.get(term, term) for term in semantics["object_terms"]),
        "direction": direction,
        "locations": frozenset(semantics["location_terms"]),
        "counts": frozenset(semantics["count_terms"]),
        "relation": relation,
        "changed": bool(semantics["changed"]),
        "no_change": bool(semantics["no_change"]),
    }


def structured_hard_negative_masks(captions: list[str], caption_to_pair: Tensor, pair_count: int) -> dict[str, Tensor]:
    """Classify deterministic caption-level structured negatives for each query/candidate pair."""
    if len(captions) != caption_to_pair.numel():
        raise ValueError("captions and caption_to_pair must have the same query count")
    candidate_captions = [""] * pair_count
    for caption, pair_index in zip(captions, caption_to_pair.tolist(), strict=True):
        if not candidate_captions[int(pair_index)]:
            candidate_captions[int(pair_index)] = caption
    queries = [_structured_signature(caption) for caption in captions]
    candidates = [_structured_signature(caption) for caption in candidate_captions]
    categories = {
        name: torch.zeros(len(captions), pair_count, dtype=torch.bool, device=caption_to_pair.device)
        for name in (
            "same_object_wrong_direction",
            "same_object_direction_wrong_location",
            "same_object_location_wrong_count",
            "same_broad_change_wrong_object",
            "no_change_lookalike",
        )
    }
    for query_index, query in enumerate(queries):
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index == int(caption_to_pair[query_index]):
                continue
            same_object = bool(query["objects"] and query["objects"] & candidate["objects"])
            same_direction = query["direction"] == candidate["direction"]
            same_location = bool(query["locations"] and query["locations"] & candidate["locations"])
            if same_object and query["direction"] not in {"unknown", "changed"} and not same_direction:
                categories["same_object_wrong_direction"][query_index, candidate_index] = True
            elif same_object and same_direction and query["locations"] and not same_location:
                categories["same_object_direction_wrong_location"][query_index, candidate_index] = True
            elif same_object and same_location and query["counts"] and query["counts"] != candidate["counts"]:
                categories["same_object_location_wrong_count"][query_index, candidate_index] = True
            elif query["changed"] and candidate["changed"] and query["objects"] and candidate["objects"] and not same_object:
                categories["same_broad_change_wrong_object"][query_index, candidate_index] = True
            elif query["changed"] and candidate["no_change"] and (same_object or not query["objects"]):
                categories["no_change_lookalike"][query_index, candidate_index] = True
    return categories


def local_positive_negative_margin_loss(
    local_scores: Tensor,
    caption_to_pair: Tensor,
    captions: list[str],
    *,
    margin: float = 0.1,
    latent_positive_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    if local_scores.ndim != 2 or local_scores.shape[0] != caption_to_pair.numel():
        raise ValueError("local_scores must have shape [Q,B]")
    categories = structured_hard_negative_masks(captions, caption_to_pair, local_scores.shape[1])
    if latent_positive_mask is not None:
        if latent_positive_mask.shape != local_scores.shape:
            raise ValueError("latent_positive_mask must match local_scores")
        categories = {name: mask & ~latent_positive_mask.to(mask.device).bool() for name, mask in categories.items()}
    combined = torch.stack(tuple(categories.values())).any(dim=0)
    rows = torch.arange(local_scores.shape[0], device=local_scores.device)
    positive = local_scores[rows, caption_to_pair.long()]
    negative = local_scores.masked_fill(~combined, float("-inf")).max(dim=1).values
    valid = torch.isfinite(negative)
    per_query = F.relu(float(margin) - positive[valid] + negative[valid])
    loss = per_query.mean() if torch.any(valid) else local_scores.sum() * 0.0
    diagnostics: dict[str, float | int] = {
        "local_margin_valid_queries": int(valid.sum().item()),
        "local_positive_negative_margin_loss": float(loss.detach().cpu()),
    }
    for name, mask in categories.items():
        diagnostics[f"hard_negative_{name}_count"] = int(mask.sum().item())
        category_negative = local_scores.masked_fill(~mask, float("-inf")).max(dim=1).values
        category_valid = torch.isfinite(category_negative)
        category_loss = (
            F.relu(float(margin) - positive[category_valid] + category_negative[category_valid]).mean()
            if torch.any(category_valid)
            else local_scores.sum() * 0.0
        )
        diagnostics[f"hard_negative_{name}_loss"] = float(category_loss.detach().cpu())
    return loss, diagnostics


def conditional_instance_discrimination_loss(
    scores: Tensor,
    caption_to_pair: Tensor,
    broad_semantic_positive_mask: Tensor,
    *,
    margin: float = 0.02,
) -> Tensor:
    """Separate exact identity only among otherwise broad-semantic positives."""
    if broad_semantic_positive_mask.shape != scores.shape:
        raise ValueError("broad_semantic_positive_mask must match scores")
    rows = torch.arange(scores.shape[0], device=scores.device)
    exact_mask = torch.zeros_like(broad_semantic_positive_mask, dtype=torch.bool)
    exact_mask[rows, caption_to_pair.long()] = True
    alternatives = broad_semantic_positive_mask.bool() & ~exact_mask
    alternative_score = scores.masked_fill(~alternatives, float("-inf")).max(dim=1).values
    valid = torch.isfinite(alternative_score)
    if not torch.any(valid):
        return scores.sum() * 0.0
    exact_score = scores[rows, caption_to_pair.long()]
    return F.relu(float(margin) - exact_score[valid] + alternative_score[valid]).mean()


def structured_auxiliary_evidence_loss(
    evidence_scores: dict[str, Tensor],
    caption_to_pair: Tensor,
    captions: list[str],
) -> tuple[Tensor, dict[str, float | int]]:
    """Supervise paired attribute evidence only for captions carrying that label."""
    signatures = [_structured_signature(caption) for caption in captions]
    specifications = {
        "object": ("S_object", "objects"),
        "direction": ("S_direction", "direction"),
        "location": ("S_location", "locations"),
        "count": ("S_count", "counts"),
        "relation": ("S_relation", "relation"),
    }
    reference = next(iter(evidence_scores.values()))
    rows = torch.arange(len(captions), device=reference.device)
    active_losses = []
    diagnostics: dict[str, float | int] = {}
    for label, (score_name, signature_key) in specifications.items():
        scores = evidence_scores[score_name]
        present = torch.tensor(
            [bool(signature[signature_key]) and signature[signature_key] not in {"unknown", "none"} for signature in signatures],
            device=scores.device,
            dtype=torch.bool,
        )
        paired = scores[rows, caption_to_pair.long()].clamp(1e-6, 1.0 - 1e-6)
        # ``paired`` is already a probability. Writing positive-label BCE as
        # -log(p) keeps the same objective while avoiding BCELoss, which CUDA
        # autocast intentionally rejects as numerically unsafe.
        loss = -paired[present].log().mean() if torch.any(present) else scores.sum() * 0.0
        if torch.any(present):
            active_losses.append(loss)
        diagnostics[f"{label}_auxiliary_label_count"] = int(present.sum().item())
        diagnostics[f"{label}_auxiliary_loss"] = float(loss.detach().cpu())
    # Attribute availability varies strongly by batch. Averaging only active
    # attributes keeps this objective on a stable scale instead of making a
    # richly annotated batch receive up to five times the gradient magnitude.
    total = torch.stack(active_losses).mean() if active_losses else reference.sum() * 0.0
    diagnostics["structured_auxiliary_active_attribute_count"] = len(active_losses)
    diagnostics["structured_auxiliary_evidence_loss"] = float(total.detach().cpu())
    return total, diagnostics


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
            initial_weights = torch.tensor([alpha, beta, max(0.05 * (alpha + beta), 1e-3)], dtype=torch.float32)
            self.fusion_logits = nn.Parameter(initial_weights.log())
            self.branch_log_scales = nn.Parameter(torch.zeros(3))
            # Per-branch constants collapse to one candidate-independent
            # offset after normalized fusion. Ranking losses are invariant to
            # that offset, so treating these values as trainable creates an
            # unidentifiable parameter with an exactly-zero real gradient.
            self.register_buffer("branch_biases", torch.zeros(3))
            for parameter in self.patch_projector.parameters():
                parameter.requires_grad_(False)
            for parameter in self.query_mask_head.parameters():
                parameter.requires_grad_(False)

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

    def score_v2(
        self,
        query_embeddings: Tensor,
        query_token_embeddings: Tensor,
        query_attention_mask: Tensor,
        pair_embeddings: Tensor,
        per_time_tokens: Tensor,
        query_metadata: dict | None = None,
    ) -> dict[str, Tensor | str]:
        if self.architecture_version != "v2":
            raise RuntimeError("score_v2 requires qcpr architecture v2")
        descriptors = F.normalize(self.temporal_descriptor_mlp(temporal_patch_descriptor(per_time_tokens)), dim=-1)
        token_embeddings = F.normalize(self.token_projection(query_token_embeddings), dim=-1)
        affinity = torch.einsum("bnd,qld->qbnl", descriptors, token_embeddings) / self.logit_scale
        affinity = affinity.masked_fill(~query_attention_mask[:, None, None, :].bool(), -1e4)
        attended_query = torch.einsum("qbnl,qld->qbnd", affinity.softmax(dim=-1), token_embeddings)
        descriptor = descriptors.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1, -1)
        temporal_explanation_logits = self.temporal_channel_head(descriptors)
        query_mask_logits = self.interaction_mlp(torch.cat((descriptor, attended_query, descriptor * attended_query), dim=-1)).squeeze(-1)
        # The interaction branch is a text-conditioned residual over learnt
        # generic-change evidence.  This prevents sparse foreground masks from
        # being minimised by an all-empty query mask while retaining text
        # selectivity in the residual. Direction-specific channels still enter
        # S_direction below rather than being conflated with generic change.
        query_mask_logits = query_mask_logits + temporal_explanation_logits[..., 0].unsqueeze(0)
        local_embeddings = self.masked_local_embedding(descriptors, query_mask_logits)
        queries = F.normalize(query_embeddings, dim=-1)
        global_score = queries @ F.normalize(pair_embeddings, dim=-1).T
        local_score = torch.einsum("qd,qbd->qb", queries, local_embeddings)
        patch_probabilities = query_mask_logits.sigmoid()
        top_k = max(1, min(4, patch_probabilities.shape[-1]))
        token_patch_score = patch_probabilities.topk(top_k, dim=-1).values.mean(dim=-1)
        raw_scores = torch.stack((global_score, local_score, token_patch_score), dim=-1)
        calibrated_scores = raw_scores * self.branch_log_scales.exp() + self.branch_biases
        fusion_weights = self.fusion_logits.softmax(dim=0)
        final_score = torch.einsum("qbk,k->qb", calibrated_scores, fusion_weights)

        metadata = query_metadata or {}
        token_group_masks = metadata.get("token_group_masks", {})

        def group_patch_score(name: str) -> Tensor:
            group_mask = token_group_masks.get(name)
            if group_mask is None:
                return global_score.new_zeros(global_score.shape)
            group_mask = group_mask.to(token_embeddings.device).bool() & query_attention_mask.bool()
            weights = group_mask.to(token_embeddings.dtype)
            pooled = torch.einsum("ql,qld->qd", weights, token_embeddings)
            pooled = F.normalize(pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0), dim=-1)
            evidence = torch.einsum("qd,bnd->qbn", pooled, descriptors).sigmoid()
            k = max(1, min(4, evidence.shape[-1]))
            score = evidence.topk(k, dim=-1).values.mean(dim=-1)
            return score * group_mask.any(dim=1).to(score.dtype)[:, None]

        object_score = group_patch_score("object")
        relation_score = group_patch_score("relation")
        direction_base = group_patch_score("direction")
        temporal_probabilities = temporal_explanation_logits.sigmoid().permute(2, 0, 1)
        direction_targets = metadata.get("direction_targets")
        direction_score = direction_base
        if direction_targets is not None:
            targets = direction_targets.to(direction_base.device).long()
            selected_channels = temporal_probabilities[0].unsqueeze(0).expand(query_embeddings.shape[0], -1, -1).clone()
            selected_channels = torch.where(
                (targets == 1)[:, None, None], temporal_probabilities[1].unsqueeze(0), selected_channels
            )
            selected_channels = torch.where(
                (targets == -1)[:, None, None], temporal_probabilities[2].unsqueeze(0), selected_channels
            )
            mask_weights = query_mask_logits.sigmoid()
            channel_evidence = (selected_channels * mask_weights).sum(dim=-1) / mask_weights.sum(dim=-1).clamp_min(1e-6)
            direction_score = direction_base * channel_evidence

        location_score = global_score.new_zeros(global_score.shape)
        location_targets = metadata.get("location_targets")
        location_mask = token_group_masks.get("location")
        if location_targets is not None and location_mask is not None:
            side = int(math.isqrt(descriptors.shape[1]))
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, side, device=descriptors.device, dtype=descriptors.dtype),
                torch.linspace(0, 1, side, device=descriptors.device, dtype=descriptors.dtype), indexing="ij",
            )
            coords = torch.stack((xx, yy), dim=-1).reshape(1, 1, -1, 2)
            targets = location_targets.to(descriptors.device, descriptors.dtype)[:, None, None, :]
            proximity = torch.exp(-4.0 * (coords - targets).square().sum(dim=-1))
            weights = query_mask_logits.sigmoid()
            location_score = (weights * proximity).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)
            location_score = location_score * location_mask.to(location_score.device).any(dim=1).to(location_score.dtype)[:, None]

        count_score = global_score.new_zeros(global_score.shape)
        count_targets = metadata.get("count_targets")
        if count_targets is not None:
            side = int(math.isqrt(query_mask_logits.shape[-1]))
            heat = query_mask_logits.sigmoid().reshape(query_mask_logits.shape[0] * query_mask_logits.shape[1], 1, side, side)
            neighborhood_max = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
            soft_peaks = heat * torch.sigmoid(20.0 * (heat - neighborhood_max)) * torch.sigmoid(10.0 * (heat - 0.5))
            distinct_regions = soft_peaks.sum(dim=(1, 2, 3)).reshape(query_mask_logits.shape[:2])
            targets = count_targets.to(distinct_regions.device, distinct_regions.dtype)
            count_score = torch.exp(-torch.abs(distinct_regions - targets[:, None]) / targets[:, None].clamp_min(1.0))
            count_score = count_score * (targets > 0).to(count_score.dtype)[:, None]
        return {
            "global_score": global_score,
            "local_score": local_score,
            "token_patch_score": token_patch_score,
            "global_score_calibrated": calibrated_scores[..., 0],
            "local_score_calibrated": calibrated_scores[..., 1],
            "token_patch_score_calibrated": calibrated_scores[..., 2],
            "fusion_weights": fusion_weights,
            "final_score": final_score,
            "S_global_raw": global_score,
            "S_local_raw": local_score,
            "S_token_patch_raw": token_patch_score,
            "S_global_calibrated": calibrated_scores[..., 0],
            "S_local_calibrated": calibrated_scores[..., 1],
            "S_token_patch_calibrated": calibrated_scores[..., 2],
            "S_final": final_score,
            "S_object": object_score,
            "S_direction": direction_score,
            "S_location": location_score,
            "S_count": count_score,
            "S_relation": relation_score,
            "query_mask_logits": query_mask_logits,
            "patch_tokens": descriptors,
            "temporal_explanation_logits": temporal_explanation_logits,
            "score_mode": "qcpr_v2",
        }


def _patch_mask_targets(pair_masks: Tensor, side: int) -> Tensor:
    """Downsample masks without silently deleting sub-patch foreground objects."""
    if pair_masks.ndim != 3:
        raise ValueError("pair_masks must have shape [B,H,W]")
    # A query-specific S2Looking object may occupy far less than one 6x6
    # descriptor cell. Nearest interpolation can map such an object entirely
    # to background, rewarding an empty query mask. Max pooling makes a
    # positive pixel visible to the descriptor cell that covers it.
    return F.adaptive_max_pool2d(pair_masks[:, None].float(), output_size=(side, side))[:, 0].flatten(1)


def _per_query_segmentation_losses(query_mask_logits: Tensor, caption_to_pair: Tensor, pair_masks: Tensor) -> Tensor:
    """Return foreground-aware BCE-plus-Dice loss for every paired query mask."""
    query_count, pair_count, patch_count = query_mask_logits.shape
    if caption_to_pair.shape != (query_count,):
        raise ValueError("caption_to_pair must contain one pair index per query")
    if pair_masks.ndim != 3 or pair_masks.shape[0] != pair_count:
        raise ValueError("pair_masks must have shape [B,H,W] with the QCPR pair count")
    side = int(math.isqrt(patch_count))
    if side * side != patch_count:
        raise ValueError("QCPR patch count must form a square grid")
    targets = _patch_mask_targets(pair_masks, side)
    rows = torch.arange(query_count, device=query_mask_logits.device)
    mapping = caption_to_pair.long()
    logits = query_mask_logits[rows, mapping]
    target = targets.to(logits.device)[mapping]
    pixel_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    foreground_count = target.sum(dim=1)
    background = 1.0 - target
    foreground_bce = (pixel_bce * target).sum(dim=1) / foreground_count.clamp_min(1.0)
    background_bce = (pixel_bce * background).sum(dim=1) / background.sum(dim=1).clamp_min(1.0)
    # Sparse non-empty masks need an explicit foreground contribution. Empty
    # targets retain the background term, so false-positive control remains
    # supervised without overwhelming the positive examples.
    bce = torch.where(
        foreground_count > 0,
        0.75 * foreground_bce + 0.25 * background_bce,
        background_bce,
    )
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
