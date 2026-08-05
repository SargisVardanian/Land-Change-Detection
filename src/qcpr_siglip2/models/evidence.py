from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.schema import Siglip2TemporalConfig


@dataclass
class EvidenceOutput:
    evidence_logits: Tensor
    evidence_weights: Tensor
    evidence_map: Tensor
    evidence_vector: Tensor
    evidence_gate: Tensor


class EvidenceBottleneck(nn.Module):
    """One text-token-to-temporal-patch evidence path used by the final score."""

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
    ) -> EvidenceOutput:
        if text_tokens.ndim != 3 or text_mask.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError("evidence inputs must be [Q,L,D], [Q,L] and [P,M,D]")
        q, l, d = text_tokens.shape
        p, m, vd = visual_tokens.shape
        if d != self.hidden_size or vd != self.hidden_size or text_mask.shape != (q, l):
            raise ValueError("evidence dimensions do not match hidden size")
        if m != frame_count * patch_count:
            raise ValueError("visual token count does not match temporal dimensions")
        valid = text_mask.bool()
        if torch.any(valid.sum(dim=1) == 0):
            raise ValueError("every query needs at least one valid text token")
        # Keep the exact operation differentiable while bounding the largest
        # temporary to Q_chunk x P_chunk x L x M.  The old implementation
        # materialized Q x P x L x M and could silently exceed H100 memory.
        text_normalized = F.normalize(text_tokens, dim=-1)
        visual_normalized = F.normalize(visual_tokens, dim=-1)
        query_chunk = min(self.query_chunk_size, q)
        pair_chunk = min(self.pair_chunk_size, p)
        logits_rows: list[Tensor] = []
        weight_rows: list[Tensor] = []
        vector_rows: list[Tensor] = []
        # ``torch.finfo(dtype).min`` is mathematically valid but passing the
        # BF16 Python float through ``masked_fill`` overflows in current
        # PyTorch. A finite sentinel is sufficient: exp(-1e4) is negligible
        # in this logsumexp while remaining representable in FP32/BF16.
        floor = -1e4
        for query_start in range(0, q, query_chunk):
            query_end = min(query_start + query_chunk, q)
            query_logits: list[Tensor] = []
            query_weights: list[Tensor] = []
            query_vectors: list[Tensor] = []
            query_mask = valid[query_start:query_end]
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
                block_logits = torch.logsumexp(similarity, dim=2)
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
        mapped = weights.reshape(q, p, frame_count, patch_count)
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError("evidence map requires square native patch grid")
        return EvidenceOutput(
            logits,
            weights,
            mapped.reshape(q, p, frame_count, side, side),
            vector,
            self.evidence_gate,
        )
