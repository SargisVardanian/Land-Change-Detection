from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QCPRV3Config:
    input_dim: int = 512
    text_dim: int = 512
    hidden_dim: int = 512
    heads: int = 8
    decoder_layers: int = 2
    scales: tuple[int, ...] = (32, 16, 8)
    output_size: tuple[int, int] = (256, 256)
    dropout: float = 0.0
    token_top_k: int = 4
    token_softmin_temperature: float = 0.1
    enable_region_slots: bool = False
    region_slots: int = 8
    slot_iterations: int = 3


@dataclass(frozen=True)
class TemporalPatchField:
    descriptors: Tensor
    coordinates: Tensor
    scale_ids: Tensor
    scale_slices: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class QCPRV3ScoreOutput:
    global_score: Tensor
    local_score: Tensor
    token_patch_score: Tensor
    reranked_score: Tensor
    patch_mask_logits: Tensor
    decoded_mask_logits: Tensor
    local_embedding: Tensor
    mask_mass: Tensor
    mask_entropy: Tensor
    mask_effective_patch_count: Tensor
    mask_validity: Tensor
    temporal_descriptors: Tensor
    grounded_patches: Tensor
    coordinates: Tensor
    scale_ids: Tensor
    temporal_map_logits: Tensor | None
    slot_activations: Tensor | None
    slot_embeddings: Tensor | None
    slot_mask_logits: Tensor | None
    memory_diagnostics: dict[str, int]


def _grid_coordinates(side: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


class ContinuousPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, frequencies: int = 4):
        super().__init__()
        self.frequencies = int(frequencies)
        input_dim = 5 + 4 * self.frequencies + 1
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coordinates: Tensor, scale: float) -> Tensor:
        x, y = coordinates.unbind(dim=-1)
        values = [x, y, x.square(), y.square(), x * y]
        for index in range(self.frequencies):
            frequency = math.pi * (2.0**index)
            values.extend((torch.sin(frequency * x), torch.cos(frequency * x)))
            values.extend((torch.sin(frequency * y), torch.cos(frequency * y)))
        values.append(torch.full_like(x, float(scale)))
        return self.projection(torch.stack(values, dim=-1))


class GenericTemporalPatchField(nn.Module):
    """Class-agnostic multi-scale field from ordered T1/T2 patch tokens."""

    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        self.position = ContinuousPositionEncoding(config.hidden_dim)
        descriptor_dim = 4 * config.input_dim
        self.descriptor_projection = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.context_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                config.heads,
                dim_feedforward=4 * config.hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(max(1, config.decoder_layers // 2))
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    @staticmethod
    def _resize_tokens(tokens: Tensor, source_side: int, target_side: int) -> Tensor:
        batch, time, _, dim = tokens.shape
        grid = tokens.reshape(batch * time, source_side, source_side, dim).permute(0, 3, 1, 2)
        if target_side == source_side:
            resized = grid
        else:
            resized = F.adaptive_avg_pool2d(grid, (target_side, target_side))
        return resized.permute(0, 2, 3, 1).reshape(batch, time, target_side * target_side, dim)

    def forward(self, per_time_tokens: Tensor) -> TemporalPatchField:
        if per_time_tokens.ndim != 4 or per_time_tokens.shape[1] != 2:
            raise ValueError("per_time_tokens must have shape [B,2,N,D]")
        if per_time_tokens.shape[-1] != self.config.input_dim:
            raise ValueError("per_time_tokens feature dimension disagrees with QCPRV3Config.input_dim")
        source_side = int(math.isqrt(per_time_tokens.shape[2]))
        if source_side * source_side != per_time_tokens.shape[2]:
            raise ValueError("per_time_tokens must form a square source grid")
        descriptors: list[Tensor] = []
        coordinates: list[Tensor] = []
        scale_ids: list[Tensor] = []
        scale_slices: list[tuple[int, int, int]] = []
        offset = 0
        for scale_index, side in enumerate(self.config.scales):
            resized = self._resize_tokens(per_time_tokens, source_side, side)
            before, after = resized[:, 0], resized[:, 1]
            raw = torch.cat((before, after, after - before, (after - before).abs()), dim=-1)
            descriptor = self.descriptor_projection(raw)
            coords = _grid_coordinates(side, device=raw.device, dtype=raw.dtype)
            descriptor = descriptor + self.position(coords, scale=float(side) / float(source_side)).unsqueeze(0)
            descriptors.append(descriptor)
            coordinates.append(coords)
            scale_ids.append(torch.full((side * side,), scale_index, device=raw.device, dtype=torch.long))
            scale_slices.append((offset, offset + side * side, side))
            offset += side * side
        field = torch.cat(descriptors, dim=1)
        for block in self.context_blocks:
            field = block(field)
        return TemporalPatchField(
            descriptors=F.normalize(self.output_norm(field), dim=-1),
            coordinates=torch.cat(coordinates, dim=0),
            scale_ids=torch.cat(scale_ids, dim=0),
            scale_slices=tuple(scale_slices),
        )


class GenericCrossModalDecoder(nn.Module):
    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        self.token_projection = nn.Linear(config.text_dim, config.hidden_dim)
        self.cross_norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(config.decoder_layers))
        self.cross_attentions = nn.ModuleList(
            nn.MultiheadAttention(config.hidden_dim, config.heads, dropout=config.dropout, batch_first=True)
            for _ in range(config.decoder_layers)
        )
        self.patch_ffns = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
                nn.GELU(),
                nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            )
            for _ in range(config.decoder_layers)
        )
        self.mask_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        patches: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if patches.ndim != 3 or text_tokens.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("patches [C,N,D], text_tokens [Q,L,D], mask [Q,L] required")
        candidates, patch_count, hidden = patches.shape
        queries, token_count, _ = text_tokens.shape
        tokens = F.normalize(self.token_projection(text_tokens), dim=-1)
        grounded = patches[None].expand(queries, -1, -1, -1).reshape(queries * candidates, patch_count, hidden)
        expanded_tokens = tokens[:, None].expand(-1, candidates, -1, -1).reshape(queries * candidates, token_count, hidden)
        expanded_mask = text_attention_mask[:, None].expand(-1, candidates, -1).reshape(queries * candidates, token_count)
        for norm, attention, ffn in zip(self.cross_norms, self.cross_attentions, self.patch_ffns, strict=True):
            attended, _ = attention(
                norm(grounded), expanded_tokens, expanded_tokens,
                key_padding_mask=~expanded_mask.bool(), need_weights=False,
            )
            grounded = grounded + attended
            grounded = grounded + ffn(grounded)
        grounded = grounded.reshape(queries, candidates, patch_count, hidden)
        patch_logits = self.mask_head(grounded).squeeze(-1)
        return grounded, patch_logits, tokens


class MultiScaleQueryMaskDecoder(nn.Module):
    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        self.refiners = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, padding_mode="replicate"), nn.GELU(),
                nn.Conv2d(16, 1, 3, padding=1, padding_mode="replicate"),
            )
            for _ in config.scales
        )
        self.output_refiner = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, padding_mode="replicate"), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, patch_logits: Tensor, scale_slices: tuple[tuple[int, int, int], ...]) -> Tensor:
        if patch_logits.ndim != 3:
            raise ValueError("patch_logits must have shape [Q,C,N]")
        queries, candidates, _ = patch_logits.shape
        target_side = max(side for _, _, side in scale_slices)
        fused = None
        for refiner, (start, end, side) in zip(self.refiners, scale_slices, strict=True):
            grid = patch_logits[..., start:end].reshape(queries * candidates, 1, side, side)
            refined = refiner(grid) + grid
            if side != target_side:
                refined = F.interpolate(refined, (target_side, target_side), mode="bilinear", align_corners=False)
            # Preserve small high-resolution evidence instead of averaging it
            # away with a coarse scale where the foreground may disappear.
            fused = refined if fused is None else torch.maximum(fused, refined)
        assert fused is not None
        fused = self.output_refiner(fused) + fused
        decoded = F.interpolate(fused, self.config.output_size, mode="bilinear", align_corners=False)
        return decoded.reshape(queries, candidates, *self.config.output_size)

    @staticmethod
    def sample_patch_logits(
        decoded_mask_logits: Tensor,
        scale_slices: tuple[tuple[int, int, int], ...],
    ) -> Tensor:
        """Sample one canonical displayed logit field back onto every patch scale."""
        if decoded_mask_logits.ndim != 4:
            raise ValueError("decoded_mask_logits must have shape [Q,C,H,W]")
        queries, candidates, height, width = decoded_mask_logits.shape
        flattened = decoded_mask_logits.reshape(queries * candidates, 1, height, width)
        sampled = [
            F.adaptive_avg_pool2d(flattened, (side, side)).reshape(queries, candidates, side * side)
            for _, _, side in scale_slices
        ]
        return torch.cat(sampled, dim=-1)


class GenericRegionSlots(nn.Module):
    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        self.slots = nn.Parameter(torch.randn(config.region_slots, config.hidden_dim) * 0.02)
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.update = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.activation = nn.Linear(config.hidden_dim, 1)

    def forward(self, grounded: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        queries, candidates, patches, dim = grounded.shape
        batch = queries * candidates
        values = grounded.reshape(batch, patches, dim)
        slots = self.slots.unsqueeze(0).expand(batch, -1, -1)
        for _ in range(self.config.slot_iterations):
            logits = torch.einsum("bkd,bnd->bkn", self.query(slots), self.key(values)) / math.sqrt(dim)
            assignments = logits.softmax(dim=1)
            updates = torch.einsum("bkn,bnd->bkd", assignments, self.value(values))
            normalizer = assignments.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            slots = self.update((updates / normalizer).reshape(-1, dim), slots.reshape(-1, dim)).reshape_as(slots)
        activations = self.activation(slots).squeeze(-1)
        logits = torch.einsum("bkd,bnd->bkn", self.query(slots), self.key(values)) / math.sqrt(dim)
        return (
            activations.reshape(queries, candidates, -1),
            slots.reshape(queries, candidates, self.config.region_slots, dim),
            logits.reshape(queries, candidates, self.config.region_slots, patches),
        )


class QCPRV3GenericGrounding(nn.Module):
    """Generic learned grounding/reranking without semantic-specific heads."""

    def __init__(self, config: QCPRV3Config | None = None):
        super().__init__()
        self.config = config or QCPRV3Config()
        self.temporal_field = GenericTemporalPatchField(self.config)
        self.grounding_decoder = GenericCrossModalDecoder(self.config)
        self.mask_decoder = MultiScaleQueryMaskDecoder(self.config)
        self.global_query_projection = (
            nn.Identity() if self.config.text_dim == self.config.hidden_dim
            else nn.Linear(self.config.text_dim, self.config.hidden_dim, bias=False)
        )
        self.local_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.temporal_map_head = nn.Linear(self.config.hidden_dim, 3)
        # B starts as global + token late interaction. Mask-local evidence is
        # effectively disabled until Stage C learns a non-empty grounded mask.
        self.rerank_logits = nn.Parameter(torch.tensor([-20.0, -2.0]))
        self.region_slots = GenericRegionSlots(self.config) if self.config.enable_region_slots else None

    @staticmethod
    def masked_local_embedding(descriptors: Tensor, mask_logits: Tensor) -> Tensor:
        weights = mask_logits.sigmoid()
        pooled = torch.einsum("qcn,cnd->qcd", weights, descriptors)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.normalize(pooled, dim=-1)

    def _token_patch_score(
        self, grounded: Tensor, projected_tokens: Tensor, attention_mask: Tensor
    ) -> Tensor:
        grounded = F.normalize(grounded, dim=-1)
        projected_tokens = F.normalize(projected_tokens, dim=-1)
        valid_rows = attention_mask.bool().any(dim=1)
        safe_mask = attention_mask.bool().clone()
        if bool((~valid_rows).any()):
            safe_mask[~valid_rows, 0] = True
        similarities = torch.einsum("qcnd,qld->qcnl", grounded, projected_tokens)
        similarities = similarities.masked_fill(~safe_mask[:, None, None, :], -1.0)
        patch_k = max(1, min(self.config.token_top_k, similarities.shape[2]))
        per_token = similarities.topk(patch_k, dim=2).values.mean(dim=2)
        valid = safe_mask[:, None].to(per_token.dtype)
        arithmetic = (per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        temperature = max(float(self.config.token_softmin_temperature), 1e-4)
        masked = per_token.masked_fill(~safe_mask[:, None], float("inf"))
        softmin = -temperature * torch.logsumexp(-masked / temperature, dim=-1)
        softmin = softmin + temperature * valid.sum(dim=-1).clamp_min(1.0).log()
        score = 0.5 * arithmetic + 0.5 * softmin
        return torch.where(valid_rows[:, None], score, torch.zeros_like(score))

    def score_query_pair_chunks(
        self,
        global_query_embeddings: Tensor,
        text_token_embeddings: Tensor,
        text_attention_mask: Tensor,
        pair_embeddings: Tensor,
        per_time_tokens: Tensor,
        *,
        text_content_mask: Tensor | None = None,
    ) -> QCPRV3ScoreOutput:
        field = self.temporal_field(per_time_tokens)
        grounded, raw_patch_logits, projected_tokens = self.grounding_decoder(
            field.descriptors, text_token_embeddings, text_attention_mask
        )
        decoded_logits = self.mask_decoder(raw_patch_logits, field.scale_slices)
        patch_logits = self.mask_decoder.sample_patch_logits(decoded_logits, field.scale_slices)
        query = F.normalize(self.global_query_projection(global_query_embeddings), dim=-1)
        pairs = F.normalize(pair_embeddings, dim=-1)
        global_score = query @ pairs.T
        local_embedding = self.masked_local_embedding(field.descriptors, patch_logits)
        probabilities = patch_logits.sigmoid()
        mask_mass = probabilities.mean(dim=-1)
        mask_peak = probabilities.amax(dim=-1)
        mask_validity = ((mask_peak - 0.10) / 0.40).clamp(0.0, 1.0)
        normalized_weights = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mask_effective_patch_count = normalized_weights.square().sum(dim=-1).clamp_min(1e-6).reciprocal()
        mask_entropy = -(
            probabilities.clamp(1e-6, 1 - 1e-6) * probabilities.clamp(1e-6, 1 - 1e-6).log()
            + (1 - probabilities).clamp(1e-6, 1 - 1e-6) * (1 - probabilities).clamp(1e-6, 1 - 1e-6).log()
        ).mean(dim=-1)
        local_score = torch.einsum("qd,qcd->qc", query, F.normalize(self.local_projection(local_embedding), dim=-1))
        local_score = local_score * mask_validity
        content_mask = text_attention_mask if text_content_mask is None else text_content_mask
        if content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask must match text_attention_mask")
        content_mask = content_mask.bool()
        fallback = ~content_mask.any(dim=1)
        if bool(fallback.any()):
            content_mask = content_mask.clone()
            content_mask[fallback] = text_attention_mask.bool()[fallback]
        token_patch_score = self._token_patch_score(grounded, projected_tokens, content_mask)
        weights = F.softplus(self.rerank_logits)
        reranked = global_score + weights[0] * local_score + weights[1] * token_patch_score
        temporal_maps = self.temporal_map_head(field.descriptors)
        slot_activations = slot_embeddings = slot_mask_logits = None
        if self.region_slots is not None:
            slot_activations, slot_embeddings, slot_mask_logits = self.region_slots(grounded)
        diagnostics = {
            "query_count": int(global_query_embeddings.shape[0]),
            "candidate_count": int(pair_embeddings.shape[0]),
            "patch_count": int(field.descriptors.shape[1]),
            "token_count": int(text_token_embeddings.shape[1]),
            "estimated_grounded_bytes": int(grounded.numel() * grounded.element_size()),
            "near_empty_mask_count": int((mask_peak < 0.10).sum().detach()),
        }
        return QCPRV3ScoreOutput(
            global_score=global_score,
            local_score=local_score,
            token_patch_score=token_patch_score,
            reranked_score=reranked,
            patch_mask_logits=patch_logits,
            decoded_mask_logits=decoded_logits,
            local_embedding=local_embedding,
            mask_mass=mask_mass,
            mask_entropy=mask_entropy,
            mask_effective_patch_count=mask_effective_patch_count,
            mask_validity=mask_validity,
            temporal_descriptors=field.descriptors,
            grounded_patches=grounded,
            coordinates=field.coordinates,
            scale_ids=field.scale_ids,
            temporal_map_logits=temporal_maps,
            slot_activations=slot_activations,
            slot_embeddings=slot_embeddings,
            slot_mask_logits=slot_mask_logits,
            memory_diagnostics=diagnostics,
        )

    def forward(self, *args: Any, **kwargs: Any) -> QCPRV3ScoreOutput:
        return self.score_query_pair_chunks(*args, **kwargs)


def stable_global_top_n(global_scores: Tensor, top_n: int) -> Tensor:
    if global_scores.ndim != 2 or top_n <= 0:
        raise ValueError("global_scores [Q,C] and positive top_n required")
    candidates = global_scores.shape[1]
    tie_break = torch.arange(candidates, device=global_scores.device, dtype=torch.float64) * -1e-12
    stable = global_scores.double() + tie_break
    return stable.topk(min(top_n, candidates), dim=1).indices


def apply_two_stage_reranking(global_scores: Tensor, reranked_scores: Tensor, top_n: int) -> Tensor:
    if global_scores.shape != reranked_scores.shape:
        raise ValueError("global and reranked score matrices must match")
    selected = stable_global_top_n(global_scores, top_n)
    output = torch.full_like(reranked_scores, float("-inf"))
    output.scatter_(1, selected, reranked_scores.gather(1, selected))
    return output


def candidate_recall_at_n(global_scores: Tensor, positive_mask: Tensor, top_n: int) -> Tensor:
    if global_scores.shape != positive_mask.shape:
        raise ValueError("positive_mask must match global_scores")
    selected = stable_global_top_n(global_scores, top_n)
    return positive_mask.bool().gather(1, selected).any(dim=1).float().mean()
