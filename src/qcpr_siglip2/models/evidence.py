from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.schema import Siglip2TemporalConfig
from ..contracts import normalize_patch_metadata


@dataclass
class EvidenceOutput:
    evidence_logits: Tensor
    evidence_weights: Tensor
    evidence_map: Tensor
    evidence_vector: Tensor
    evidence_score: Tensor
    evidence_gate: Tensor
    evidence_map_valid_mask: Tensor
    processed_evidence_map: Tensor | None = None
    processed_evidence_map_valid_mask: Tensor | None = None


class EvidenceBottleneck(nn.Module):
    """Query-conditioned evidence over padded, variable-resolution tokens."""

    def __init__(self, config: Siglip2TemporalConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.evidence_temperature = float(config.evidence_temperature)
        self.query_chunk_size = int(config.evidence_query_chunk_size)
        self.pair_chunk_size = int(config.evidence_pair_chunk_size)
        self.raw_gate = nn.Parameter(torch.tensor(config.evidence_gate_raw_init))

    @property
    def evidence_gate(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.raw_gate)

    def forward(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        visual_tokens: Tensor,
        *,
        frame_count: int,
        patch_count: int,
        patch_valid_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        native_patch_valid_mask: Tensor | None = None,
        native_spatial_shapes: Tensor | None = None,
        region_assignment: Tensor | None = None,
    ) -> EvidenceOutput:
        if text_tokens.ndim != 3 or text_mask.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError("evidence inputs must be [Q,L,D], [Q,L] and [P,M,D]")
        q, l, d = text_tokens.shape
        p, m, vd = visual_tokens.shape
        if d != self.hidden_size or vd != self.hidden_size or text_mask.shape != (q, l):
            raise ValueError("evidence dimensions do not match hidden size")
        if m != frame_count * patch_count:
            raise ValueError("visual token count does not match temporal dimensions")
        valid_text = text_mask.bool()
        if torch.any(valid_text.sum(dim=1) == 0):
            raise ValueError("every query needs at least one valid text token")

        visual_view = visual_tokens.reshape(p, frame_count, patch_count, vd)
        visual_valid, visual_shapes = normalize_patch_metadata(
            visual_view, patch_valid_mask, spatial_shapes
        )
        flat_visual_valid = visual_valid.reshape(p, m)
        # Keep the exact operation differentiable while bounding the largest
        # temporary to Q_chunk x P_chunk x L x M.
        text_normalized = F.normalize(text_tokens, dim=-1)
        visual_normalized = F.normalize(visual_tokens, dim=-1)
        query_chunk = min(self.query_chunk_size, q)
        pair_chunk = min(self.pair_chunk_size, p)
        logits_rows: list[Tensor] = []
        weight_rows: list[Tensor] = []
        vector_rows: list[Tensor] = []
        # BF16 cannot represent the Python float produced by finfo.min in all
        # masked_fill paths.  This finite sentinel is still far outside the
        # useful similarity range and keeps the operation stable.
        floor = -1e4
        for query_start in range(0, q, query_chunk):
            query_end = min(query_start + query_chunk, q)
            query_logits: list[Tensor] = []
            query_weights: list[Tensor] = []
            query_vectors: list[Tensor] = []
            query_mask = valid_text[query_start:query_end]
            for pair_start in range(0, p, pair_chunk):
                pair_end = min(pair_start + pair_chunk, p)
                similarity = torch.einsum(
                    "qld,pmd->qplm",
                    text_normalized[query_start:query_end],
                    visual_normalized[pair_start:pair_end],
                )
                similarity = similarity.masked_fill(
                    ~query_mask[:, None, :, None], floor
                )
                valid_token_count = query_mask.sum(dim=1).to(similarity.dtype)
                block_logits = torch.logsumexp(similarity, dim=2) - valid_token_count.log()[:, None, None]
                block_logits = block_logits.masked_fill(
                    ~flat_visual_valid[pair_start:pair_end][None, :, :], floor
                )
                block_weights = torch.softmax(
                    block_logits / self.evidence_temperature, dim=-1
                )
                block_vector = torch.einsum(
                    "qpm,pmd->qpd",
                    block_weights,
                    visual_tokens[pair_start:pair_end],
                )
                query_logits.append(block_logits)
                query_weights.append(block_weights)
                query_vectors.append(block_vector)
            logits_rows.append(torch.cat(query_logits, dim=1))
            weight_rows.append(torch.cat(query_weights, dim=1))
            vector_rows.append(torch.cat(query_vectors, dim=1))
        logits = torch.cat(logits_rows, dim=0)
        weights = torch.cat(weight_rows, dim=0)
        vector = torch.cat(vector_rows, dim=0)
        evidence_score = (weights * logits).sum(dim=-1)

        processed_max_height = int(visual_shapes[..., 0].max().item())
        processed_max_width = int(visual_shapes[..., 1].max().item())
        processed_mapped = torch.zeros(
            q,
            p,
            frame_count,
            processed_max_height,
            processed_max_width,
            device=weights.device,
            dtype=weights.dtype,
        )
        processed_map_valid = torch.zeros(
            p,
            frame_count,
            processed_max_height,
            processed_max_width,
            device=weights.device,
            dtype=torch.bool,
        )
        weight_view = weights.reshape(q, p, frame_count, patch_count)
        for pair_index in range(p):
            for frame_index in range(frame_count):
                height = int(visual_shapes[pair_index, frame_index, 0].item())
                width = int(visual_shapes[pair_index, frame_index, 1].item())
                count = height * width
                processed_mapped[
                    :, pair_index, frame_index, :height, :width
                ] = weight_view[
                    :, pair_index, frame_index, :count
                ].reshape(q, height, width)
                processed_map_valid[pair_index, frame_index, :height, :width] = True

        native_valid = visual_valid
        native_shapes = visual_shapes
        if native_patch_valid_mask is not None or native_spatial_shapes is not None:
            if native_patch_valid_mask is not None:
                native_token_count = int(native_patch_valid_mask.shape[-1])
            elif native_spatial_shapes is not None:
                native_token_count = int(
                    (native_spatial_shapes[..., 0] * native_spatial_shapes[..., 1])
                    .max()
                    .item()
                )
            else:
                native_token_count = patch_count
            native_view = visual_tokens.new_zeros(
                p,
                frame_count,
                native_token_count,
                self.hidden_size,
            )
            native_valid, native_shapes = normalize_patch_metadata(
                native_view,
                native_patch_valid_mask,
                native_spatial_shapes,
            )
        if region_assignment is not None:
            if region_assignment.ndim != 4 or region_assignment.shape[:3] != (
                p,
                frame_count,
                patch_count,
            ):
                raise ValueError(
                    "region_assignment must be [P,T,processed_N,native_N]"
                )
            if region_assignment.shape[-1] != native_valid.shape[-1]:
                raise ValueError("region_assignment native token count mismatch")
            assignment = region_assignment.to(device=weights.device, dtype=weights.dtype)
            assignment = assignment.masked_fill(~native_valid[:, :, None, :], 0.0)
            assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            native_weight_view = torch.einsum(
                "qptk,ptkn->qptn", weight_view, assignment
            )
        else:
            if native_valid.shape[-1] != patch_count:
                raise ValueError(
                    "native and processed token counts differ without region_assignment"
                )
            native_weight_view = weight_view

        native_max_height = int(native_shapes[..., 0].max().item())
        native_max_width = int(native_shapes[..., 1].max().item())
        mapped = torch.zeros(
            q,
            p,
            frame_count,
            native_max_height,
            native_max_width,
            device=weights.device,
            dtype=weights.dtype,
        )
        map_valid = torch.zeros(
            p,
            frame_count,
            native_max_height,
            native_max_width,
            device=weights.device,
            dtype=torch.bool,
        )
        for pair_index in range(p):
            for frame_index in range(frame_count):
                height = int(native_shapes[pair_index, frame_index, 0].item())
                width = int(native_shapes[pair_index, frame_index, 1].item())
                count = height * width
                mapped[:, pair_index, frame_index, :height, :width] = native_weight_view[
                    :, pair_index, frame_index, :count
                ].reshape(q, height, width)
                map_valid[pair_index, frame_index, :height, :width] = True
        return EvidenceOutput(
            evidence_logits=logits,
            evidence_weights=weights,
            evidence_map=mapped,
            evidence_vector=vector,
            evidence_score=evidence_score,
            evidence_gate=self.evidence_gate,
            evidence_map_valid_mask=map_valid,
            processed_evidence_map=processed_mapped,
            processed_evidence_map_valid_mask=processed_map_valid,
        )
