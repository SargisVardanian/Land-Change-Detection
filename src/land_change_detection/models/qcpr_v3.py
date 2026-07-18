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
    visual_source_dim: int | None = None
    text_dim: int = 512
    global_text_dim: int | None = None
    hidden_dim: int = 256
    heads: int = 8
    decoder_layers: int = 1
    scales: tuple[int, ...] = ()
    output_size: tuple[int, int] = (256, 256)
    dropout: float = 0.0
    token_top_k: int = 4
    token_softmin_temperature: float = 0.1
    enable_region_slots: bool = False
    region_slots: int = 8
    slot_iterations: int = 3
    mask_decoder_dim: int = 64
    mask_prior_probability: float = 0.05


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


@dataclass(frozen=True)
class QCPRV3AlignedMaskOutput:
    """One decoded mask per query for its known supervised physical pair."""

    patch_mask_logits: Tensor
    decoded_mask_logits: Tensor
    local_embedding: Tensor
    temporal_descriptors: Tensor
    matched_temporal_descriptors: Tensor
    grounded_patches: Tensor
    projected_tokens: Tensor
    coordinates: Tensor
    scale_ids: Tensor
    memory_diagnostics: dict[str, int]


def _grid_coordinates(side: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def _fixed_2d_position(coordinates: Tensor, hidden_dim: int) -> Tensor:
    """Deterministic Fourier position encoding with no learned location rules."""
    quarter = max(1, hidden_dim // 4)
    frequencies = torch.exp(
        torch.linspace(0.0, math.log(100.0), quarter, device=coordinates.device, dtype=coordinates.dtype)
    )
    x, y = coordinates.unbind(dim=-1)
    values = torch.cat(
        (
            torch.sin(x[:, None] * frequencies[None]),
            torch.cos(x[:, None] * frequencies[None]),
            torch.sin(y[:, None] * frequencies[None]),
            torch.cos(y[:, None] * frequencies[None]),
        ),
        dim=-1,
    )
    if values.shape[-1] < hidden_dim:
        values = F.pad(values, (0, hidden_dim - values.shape[-1]))
    return values[:, :hidden_dim]


class GenericTemporalPatchField(nn.Module):
    """One native class-agnostic field from ordered T1/T2 patch tokens."""

    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        source_dim = config.input_dim if config.visual_source_dim is None else config.visual_source_dim
        self.source_projection = (
            nn.Identity() if source_dim == config.input_dim
            else nn.Linear(source_dim, config.input_dim)
        )
        self.context_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.delta_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.magnitude_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, per_time_tokens: Tensor) -> TemporalPatchField:
        if per_time_tokens.ndim != 4 or per_time_tokens.shape[1] != 2:
            raise ValueError("per_time_tokens must have shape [B,2,N,D]")
        source_dim = self.config.input_dim if self.config.visual_source_dim is None else self.config.visual_source_dim
        if per_time_tokens.shape[-1] != source_dim:
            raise ValueError("per_time_tokens feature dimension disagrees with QCPRV3Config.visual_source_dim")
        per_time_tokens = self.source_projection(per_time_tokens)
        source_side = int(math.isqrt(per_time_tokens.shape[2]))
        if source_side * source_side != per_time_tokens.shape[2]:
            raise ValueError("per_time_tokens must form a square source grid")
        before, after = per_time_tokens[:, 0], per_time_tokens[:, 1]
        context = 0.5 * (before + after)
        delta = after - before
        magnitude = delta.abs()
        coords = _grid_coordinates(source_side, device=before.device, dtype=before.dtype)
        field = (
            self.context_projection(context)
            + self.delta_projection(delta)
            + self.magnitude_projection(magnitude)
            + _fixed_2d_position(coords, self.config.hidden_dim).unsqueeze(0)
        )
        return TemporalPatchField(
            descriptors=F.normalize(self.output_norm(field), dim=-1),
            coordinates=coords,
            scale_ids=torch.zeros(source_side * source_side, device=before.device, dtype=torch.long),
            scale_slices=((0, source_side * source_side, source_side),),
        )


class GenericCrossModalDecoder(nn.Module):
    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        self.token_projection = nn.Linear(config.text_dim, config.hidden_dim)
        self.query_modulation = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
        )
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
        global_query_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if patches.ndim != 3 or text_tokens.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("patches [C,N,D], text_tokens [Q,L,D], mask [Q,L] required")
        candidates, patch_count, hidden = patches.shape
        queries, token_count, _ = text_tokens.shape
        tokens = F.normalize(self.token_projection(text_tokens), dim=-1)
        valid = text_attention_mask.to(tokens.dtype).unsqueeze(-1)
        query_context = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if global_query_embedding is not None:
            if global_query_embedding.shape != query_context.shape:
                raise ValueError("global query embedding must align with projected token context")
            query_context = F.normalize(query_context, dim=-1) + F.normalize(global_query_embedding, dim=-1)
        scale, shift = self.query_modulation(query_context).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)
        grounded = patches[None].expand(queries, -1, -1, -1)
        grounded = grounded * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        grounded = grounded.reshape(queries * candidates, patch_count, hidden)
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

    def forward_aligned(
        self,
        patches: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
        global_query_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Ground Q queries against Q already-matched patch fields."""
        if patches.ndim != 3 or text_tokens.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("aligned patches [Q,N,D], text_tokens [Q,L,D], mask [Q,L] required")
        queries, _, hidden = patches.shape
        if text_tokens.shape[0] != queries or text_attention_mask.shape[0] != queries:
            raise ValueError("aligned query and patch batch dimensions must match")
        if hidden != self.config.hidden_dim:
            raise ValueError("aligned patch hidden dimension does not match decoder")
        tokens = F.normalize(self.token_projection(text_tokens), dim=-1)
        valid = text_attention_mask.to(tokens.dtype).unsqueeze(-1)
        query_context = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if global_query_embedding is not None:
            if global_query_embedding.shape != query_context.shape:
                raise ValueError("global query embedding must align with projected token context")
            query_context = F.normalize(query_context, dim=-1) + F.normalize(global_query_embedding, dim=-1)
        scale, shift = self.query_modulation(query_context).chunk(2, dim=-1)
        grounded = patches * (1.0 + 0.5 * torch.tanh(scale)[:, None, :]) + shift[:, None, :]
        for norm, attention, ffn in zip(self.cross_norms, self.cross_attentions, self.patch_ffns, strict=True):
            attended, _ = attention(
                norm(grounded), tokens, tokens,
                key_padding_mask=~text_attention_mask.bool(), need_weights=False,
            )
            grounded = grounded + attended
            grounded = grounded + ffn(grounded)
        patch_logits = self.mask_head(grounded).squeeze(-1)
        return grounded, patch_logits, tokens


class MultiScaleQueryMaskDecoder(nn.Module):
    def __init__(self, config: QCPRV3Config):
        super().__init__()
        self.config = config
        if not 0.0 < config.mask_prior_probability < 1.0:
            raise ValueError("mask_prior_probability must be strictly between zero and one")
        width = config.mask_decoder_dim
        self.lateral_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, width))
            for _ in range(3)
        )
        self.logit_projections = nn.ModuleList(nn.Conv2d(1, width, 1) for _ in range(3))
        self.fusion_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
                nn.GroupNorm(8, width), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"), nn.GELU(),
            )
            for _ in range(3)
        )
        self.upsample_refinement = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        # Sparse EO targets should not begin as an all-foreground prediction.
        # The prior only initializes the generic final logit bias; it does not
        # encode any object, location, count, or temporal-direction rule.
        nn.init.constant_(
            self.mask_head[-1].bias,
            math.log(config.mask_prior_probability / (1.0 - config.mask_prior_probability)),
        )

    def forward(self, patch_logits: Tensor, grounded: Tensor, scale_slices: tuple[tuple[int, int, int], ...]) -> Tensor:
        if patch_logits.ndim != 3 or grounded.ndim != 4 or grounded.shape[:3] != patch_logits.shape:
            raise ValueError("patch_logits [Q,C,N] and grounded [Q,C,N,D] must align")
        queries, candidates, _ = patch_logits.shape
        if len(scale_slices) != 1:
            raise ValueError("mask decoder requires one native temporal patch grid")
        start, end, native_side = scale_slices[0]
        native_features = grounded[..., start:end, :].reshape(
            queries * candidates, native_side, native_side, -1
        ).permute(0, 3, 1, 2)
        native_logits = patch_logits[..., start:end].reshape(
            queries * candidates, 1, native_side, native_side
        )
        lateral: list[Tensor] = []
        for level, (projection, logit_projection) in enumerate(
            zip(self.lateral_projections, self.logit_projections, strict=True)
        ):
            side = max(1, native_side // (2**level))
            features = F.adaptive_avg_pool2d(native_features, (side, side))
            features = projection(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            logits = F.adaptive_avg_pool2d(native_logits, (side, side))
            lateral.append(features + logit_projection(logits))
        if not lateral:
            raise ValueError("at least one mask scale is required")
        fused: Tensor | None = None
        for index in reversed(range(len(lateral))):
            current = lateral[index]
            if fused is not None:
                fused = F.interpolate(fused, current.shape[-2:], mode="bilinear", align_corners=False)
                current = current + fused
            fused = self.fusion_blocks[index](current)
        assert fused is not None and fused.shape[-2:] == (native_side, native_side)
        fused = F.interpolate(
            fused, size=self.config.output_size, mode="bilinear", align_corners=False
        )
        fused = self.upsample_refinement(fused)
        decoded = self.mask_head(fused)
        if decoded.shape[-2:] != self.config.output_size:
            raise RuntimeError(
                f"learned mask decoder produced {tuple(decoded.shape[-2:])}, expected {self.config.output_size}"
            )
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
        self.direct_token_projection = nn.Linear(self.config.text_dim, self.config.hidden_dim)
        self.grounding_decoder = GenericCrossModalDecoder(self.config)
        self.mask_decoder = MultiScaleQueryMaskDecoder(self.config)
        global_text_dim = (
            self.config.text_dim
            if self.config.global_text_dim is None
            else self.config.global_text_dim
        )
        self.global_query_projection = (
            nn.Identity() if global_text_dim == self.config.hidden_dim
            else nn.Linear(global_text_dim, self.config.hidden_dim, bias=False)
        )
        self.local_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

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
        score = (per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
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
        decode_mask: bool = True,
    ) -> QCPRV3ScoreOutput:
        field = self.temporal_field(per_time_tokens)
        query_condition = self.global_query_projection(global_query_embeddings)
        global_query = F.normalize(global_query_embeddings, dim=-1)
        query = F.normalize(query_condition, dim=-1)
        pairs = F.normalize(pair_embeddings, dim=-1)
        if global_query.shape[-1] != pairs.shape[-1]:
            raise ValueError("global query and pair embedding dimensions must match")
        global_score = global_query @ pairs.T
        content_mask = text_attention_mask if text_content_mask is None else text_content_mask
        if content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask must match text_attention_mask")
        content_mask = content_mask.bool()
        fallback = ~content_mask.any(dim=1)
        if bool(fallback.any()):
            content_mask = content_mask.clone()
            content_mask[fallback] = text_attention_mask.bool()[fallback]
        direct_tokens = self.direct_token_projection(text_token_embeddings)
        direct_patches = field.descriptors[None].expand(
            global_query_embeddings.shape[0], -1, -1, -1
        )
        token_patch_score = self._token_patch_score(
            direct_patches, direct_tokens, content_mask
        )
        reranked = global_score + 0.1 * token_patch_score
        if decode_mask:
            grounded, raw_patch_logits, _ = self.grounding_decoder(
                field.descriptors, text_token_embeddings, text_attention_mask,
                query_condition,
            )
            decoded_logits = self.mask_decoder(raw_patch_logits, grounded, field.scale_slices)
            patch_logits = self.mask_decoder.sample_patch_logits(decoded_logits, field.scale_slices)
            local_embedding = self.masked_local_embedding(field.descriptors, patch_logits)
        else:
            queries, candidates = global_score.shape
            grounded = field.descriptors.new_empty(queries, candidates, 0, self.config.hidden_dim)
            decoded_logits = field.descriptors.new_empty(queries, candidates, 0, 0)
            patch_logits = field.descriptors.new_empty(queries, candidates, 0)
            local_embedding = field.descriptors.new_zeros(queries, candidates, self.config.hidden_dim)
        probabilities = patch_logits.sigmoid()
        mask_mass = probabilities.mean(dim=-1) if probabilities.numel() else global_score.new_zeros(global_score.shape)
        mask_peak = probabilities.amax(dim=-1) if probabilities.numel() else global_score.new_zeros(global_score.shape)
        mask_validity = ((mask_peak - 0.10) / 0.40).clamp(0.0, 1.0)
        normalized_weights = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mask_effective_patch_count = (
            normalized_weights.square().sum(dim=-1).clamp_min(1e-6).reciprocal()
            if probabilities.numel() else global_score.new_zeros(global_score.shape)
        )
        mask_entropy = -(
            probabilities.clamp(1e-6, 1 - 1e-6) * probabilities.clamp(1e-6, 1 - 1e-6).log()
            + (1 - probabilities).clamp(1e-6, 1 - 1e-6) * (1 - probabilities).clamp(1e-6, 1 - 1e-6).log()
        ).mean(dim=-1) if probabilities.numel() else global_score.new_zeros(global_score.shape)
        local_score = torch.einsum("qd,qcd->qc", query, F.normalize(self.local_projection(local_embedding), dim=-1))
        local_score = local_score * mask_validity
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
            temporal_map_logits=None,
            slot_activations=None,
            slot_embeddings=None,
            slot_mask_logits=None,
            memory_diagnostics=diagnostics,
        )

    def score_aligned_query_pairs(
        self,
        global_query_embeddings: Tensor,
        text_token_embeddings: Tensor,
        text_attention_mask: Tensor,
        per_time_tokens: Tensor,
        query_to_pair: Tensor,
    ) -> QCPRV3AlignedMaskOutput:
        """Decode exactly Q supervised masks for Q directional queries."""
        field = self.temporal_field(per_time_tokens)
        if query_to_pair.ndim != 1 or query_to_pair.numel() != global_query_embeddings.shape[0]:
            raise ValueError("query_to_pair must be rank one with one entry per query")
        query_to_pair = query_to_pair.to(device=field.descriptors.device, dtype=torch.long)
        if query_to_pair.numel() and (
            int(query_to_pair.min()) < 0 or int(query_to_pair.max()) >= field.descriptors.shape[0]
        ):
            raise ValueError("query_to_pair contains an out-of-range physical pair")
        matched = field.descriptors.index_select(0, query_to_pair)
        query_condition = self.global_query_projection(global_query_embeddings)
        grounded, raw_patch_logits, projected_tokens = self.grounding_decoder.forward_aligned(
            matched,
            text_token_embeddings,
            text_attention_mask,
            query_condition,
        )
        decoded = self.mask_decoder(
            raw_patch_logits[:, None], grounded[:, None], field.scale_slices
        )[:, 0]
        patch_logits = self.mask_decoder.sample_patch_logits(
            decoded[:, None], field.scale_slices
        )[:, 0]
        local_embedding = self.masked_local_embedding(
            matched, patch_logits[:, None]
        )[:, 0]
        diagnostics = {
            "physical_pair_count": int(field.descriptors.shape[0]),
            "directional_query_count": int(query_to_pair.numel()),
            "decoded_mask_count": int(decoded.shape[0]),
            "cartesian_mask_count": 0,
            "patch_count": int(field.descriptors.shape[1]),
            "estimated_grounded_bytes": int(grounded.numel() * grounded.element_size()),
        }
        return QCPRV3AlignedMaskOutput(
            patch_mask_logits=patch_logits,
            decoded_mask_logits=decoded,
            local_embedding=local_embedding,
            temporal_descriptors=field.descriptors,
            matched_temporal_descriptors=matched,
            grounded_patches=grounded,
            projected_tokens=projected_tokens,
            coordinates=field.coordinates,
            scale_ids=field.scale_ids,
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
