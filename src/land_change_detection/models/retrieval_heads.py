from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalHeadOutput:
    pair_embedding: Tensor
    text_embedding: Tensor | None
    logits: Tensor | None


class RetrievalProjectionHead(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        *,
        trainable_temperature: bool = False,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if max_logit_scale <= 0.0:
            raise ValueError("max_logit_scale must be positive")
        self.pair_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        initial_log_scale = math.log(1.0 / initial_temperature)
        log_scale = torch.tensor(initial_log_scale, dtype=torch.float32)
        if trainable_temperature:
            self.logit_scale: nn.Parameter | None = nn.Parameter(log_scale)
            self.fixed_logit_scale: float | None = None
        else:
            # Keep legacy state_dict compatibility: a non-trainable temperature
            # is a plain scalar, not a persistent buffer/parameter.
            self.register_parameter("logit_scale", None)
            self.fixed_logit_scale = float(1.0 / initial_temperature)
        self.max_logit_scale = float(max_logit_scale)

    def similarity_scale(self) -> Tensor:
        if self.logit_scale is None:
            value = min(float(self.fixed_logit_scale), self.max_logit_scale)
            return self.pair_projection[1].weight.new_tensor(value)
        # Smooth upper bound keeps gradients alive near the configured maximum.
        max_log_scale = math.log(self.max_logit_scale)
        raw = self.logit_scale
        beta = 10.0
        lower_bounded = F.softplus(raw * beta) / beta
        bounded = max_log_scale - F.softplus((max_log_scale - lower_bounded) * beta) / beta
        return bounded.exp()

    def forward(self, pair_embedding: Tensor, text_embedding: Tensor | None = None) -> RetrievalHeadOutput:
        pair = F.normalize(self.pair_projection(pair_embedding), dim=-1)
        text = F.normalize(text_embedding, dim=-1) if text_embedding is not None else None
        # Keep unscaled logits for backward compatibility. The training objective
        # consumes ``similarity_scale()`` explicitly.
        logits = pair @ text.T if text is not None else None
        return RetrievalHeadOutput(pair_embedding=pair, text_embedding=text, logits=logits)


class TextEmbeddingAdapter(nn.Module):
    """Near-identity residual adapter for frozen text embeddings."""

    def __init__(self, dim: int = 512, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_dim or dim)
        self.norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, embeddings: Tensor) -> Tensor:
        adapted = embeddings + self.adapter(self.norm(embeddings))
        return F.normalize(adapted, dim=-1)


def normalize_caption_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(normalized.split())


def classify_caption_semantics(text: str) -> dict[str, bool]:
    """Central lightweight caption semantics for audit strata and filtering."""

    normalized = normalize_caption_text(text)
    tokens = set(normalized.split())
    no_change = (
        "no change" in normalized
        or "unchanged" in tokens
        or "without change" in normalized
        or "same" in tokens
    )
    appeared = any(term in tokens for term in ("appeared", "appears", "built", "new", "added", "emerged"))
    disappeared = any(term in tokens for term in ("disappeared", "disappears", "removed", "lost", "demolished", "gone"))
    changed = (
        not no_change
        and (
            "change" in tokens
            or "changed" in tokens
            or appeared
            or disappeared
            or any(term in tokens for term in ("increased", "decreased", "expanded", "reduced"))
        )
    )
    return {
        "no_change": bool(no_change and not (appeared or disappeared)),
        "changed": bool(changed or appeared or disappeared),
        "appeared": bool(appeared),
        "disappeared": bool(disappeared),
    }


def stable_caption_group_ids(captions: list[str], device: torch.device | str | None = None) -> Tensor:
    values: list[int] = []
    for caption in captions:
        normalized = normalize_caption_text(caption)
        digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
        values.append(int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFF_FFFF_FFFF_FFFF)
    return torch.tensor(values, dtype=torch.long, device=device)


def build_caption_positive_mask(
    caption_to_pair: Tensor,
    pair_count: int,
    caption_group_ids: Tensor | None = None,
) -> Tensor:
    mapping = caption_to_pair.long()
    if mapping.ndim != 1:
        raise ValueError(f"caption_to_pair must be one-dimensional, got {tuple(mapping.shape)}")
    if mapping.numel() == 0:
        raise ValueError("At least one caption is required for retrieval loss.")
    if int(mapping.min().item()) < 0 or int(mapping.max().item()) >= pair_count:
        raise ValueError("caption_to_pair contains an out-of-range pair index.")

    caption_count = mapping.shape[0]
    positives = torch.zeros(pair_count, caption_count, dtype=torch.bool, device=mapping.device)
    columns = torch.arange(caption_count, device=mapping.device)
    positives[mapping, columns] = True

    if caption_group_ids is None:
        return positives

    groups = caption_group_ids.to(device=mapping.device, dtype=torch.long)
    if groups.shape != mapping.shape:
        raise ValueError(
            f"caption_group_ids must have shape {tuple(mapping.shape)}, got {tuple(groups.shape)}"
        )

    same_group = groups[:, None] == groups[None, :]
    caption_pair_matrix = torch.zeros(caption_count, pair_count, dtype=torch.bool, device=mapping.device)
    caption_pair_matrix[columns, mapping] = True
    group_relevant_pairs = same_group.float() @ caption_pair_matrix.float()
    positives |= group_relevant_pairs.T.to(torch.bool)
    return positives


def _infer_duplicate_groups_from_embeddings(text_embeddings: Tensor) -> Tensor:
    detached = F.normalize(text_embeddings.detach().float(), dim=-1)
    same_text = torch.isclose(
        detached[:, None, :],
        detached[None, :, :],
        rtol=1e-5,
        atol=1e-6,
    ).all(dim=-1)
    group_ids = torch.full(
        (detached.shape[0],),
        -1,
        dtype=torch.long,
        device=detached.device,
    )
    next_group = 0
    for caption_index in range(detached.shape[0]):
        if group_ids[caption_index] >= 0:
            continue
        members = torch.nonzero(same_text[caption_index], as_tuple=False).flatten()
        group_ids[members] = next_group
        next_group += 1
    return group_ids


def _scaled_similarity_logits(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    *,
    temperature: float,
    logit_scale: Tensor | float | None,
) -> Tensor:
    pair = F.normalize(pair_embeddings, dim=-1)
    text = F.normalize(text_embeddings, dim=-1)
    if logit_scale is None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return pair @ text.T / temperature
    scale = torch.as_tensor(logit_scale, dtype=pair.dtype, device=pair.device)
    if scale.numel() != 1:
        raise ValueError("logit_scale must be scalar")
    return pair @ text.T * scale


def multi_positive_symmetric_info_nce(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    caption_group_ids: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    logits = _scaled_similarity_logits(
        pair_embeddings,
        text_embeddings,
        temperature=temperature,
        logit_scale=None,
    )
    groups = caption_group_ids
    if groups is None:
        groups = _infer_duplicate_groups_from_embeddings(text_embeddings)
    positives = build_caption_positive_mask(
        caption_to_pair,
        pair_count=pair_embeddings.shape[0],
        caption_group_ids=groups,
    )
    pair_log_prob = logits.log_softmax(dim=1)
    text_log_prob = logits.T.log_softmax(dim=1)
    pair_loss = -(pair_log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1)).mean()
    text_positives = positives.T
    text_loss = -(text_log_prob.masked_fill(~text_positives, 0.0).sum(dim=1) / text_positives.sum(dim=1).clamp_min(1)).mean()
    return 0.5 * (pair_loss + text_loss)


def _positive_set_mass_loss(logits: Tensor, positives: Tensor) -> Tensor:
    if logits.shape != positives.shape:
        raise ValueError(f"logits and positives must have the same shape, got {logits.shape} and {positives.shape}")
    valid = positives.any(dim=1)
    if not torch.all(valid):
        raise ValueError("Every query must have at least one positive target")
    logits32 = logits.float()
    positive_logits = logits32.masked_fill(~positives, float("-inf"))
    return (torch.logsumexp(logits32, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def multi_positive_set_info_nce(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    temperature: float = 0.07,
    logit_scale: Tensor | float | None = None,
    text_to_pair_weight: float = 0.75,
    pair_to_text_weight: float = 0.25,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Set-mass contrastive objective aligned with duplicate-aware retrieval.

    Unlike averaging ``-log p`` over every positive, this objective maximizes
    the probability mass assigned to the positive *set*. It therefore does not
    impose an artificial ``log(number_of_positives)`` floor when one query has
    several valid targets.
    """

    if text_to_pair_weight < 0.0 or pair_to_text_weight < 0.0:
        raise ValueError("loss direction weights must be non-negative")
    weight_sum = text_to_pair_weight + pair_to_text_weight
    if weight_sum <= 0.0:
        raise ValueError("at least one loss direction weight must be positive")

    logits = _scaled_similarity_logits(
        pair_embeddings,
        text_embeddings,
        temperature=temperature,
        logit_scale=logit_scale,
    )
    positives = build_caption_positive_mask(
        caption_to_pair,
        pair_count=pair_embeddings.shape[0],
        caption_group_ids=caption_group_ids,
    )
    pair_to_text = _positive_set_mass_loss(logits, positives)
    text_to_pair = _positive_set_mass_loss(logits.T, positives.T)
    loss = (
        text_to_pair_weight * text_to_pair
        + pair_to_text_weight * pair_to_text
    ) / weight_sum
    if not return_diagnostics:
        return loss
    return loss, {
        "text_to_pair_loss": float(text_to_pair.detach().cpu()),
        "pair_to_text_loss": float(pair_to_text.detach().cpu()),
        "text_to_pair_weight": float(text_to_pair_weight / weight_sum),
        "pair_to_text_weight": float(pair_to_text_weight / weight_sum),
        "positive_pairs": int(positives.sum().detach().cpu()),
        "min_text_positives": int(positives.T.sum(dim=1).min().detach().cpu()),
        "min_pair_positives": int(positives.sum(dim=1).min().detach().cpu()),
    }


class FalseNegativeSafeEmbeddingQueue:
    """Detached CPU queue for optional global negatives with caption-group guards."""

    def __init__(self, max_size: int = 0, dim: int | None = None):
        self.max_size = int(max_size)
        self.dim = dim
        self.embeddings = torch.empty(0, dim or 0, dtype=torch.float32)
        self.group_ids = torch.empty(0, dtype=torch.long)

    def reset(self) -> None:
        self.embeddings = torch.empty(0, self.dim or 0, dtype=torch.float32)
        self.group_ids = torch.empty(0, dtype=torch.long)

    def enqueue(self, embeddings: Tensor, group_ids: Tensor) -> None:
        if self.max_size <= 0 or embeddings.numel() == 0:
            return
        detached = F.normalize(embeddings.detach().float().cpu(), dim=-1)
        groups = group_ids.detach().long().cpu()
        if self.dim is None:
            self.dim = int(detached.shape[-1])
        if detached.shape[-1] != self.dim:
            raise ValueError(f"Queue embedding dim mismatch: expected {self.dim}, got {detached.shape[-1]}")
        self.embeddings = torch.cat([self.embeddings.to(detached), detached], dim=0)[-self.max_size :]
        self.group_ids = torch.cat([self.group_ids, groups], dim=0)[-self.max_size :]

    def safe_negative_mask(self, query_group_ids: Tensor) -> Tensor:
        if self.group_ids.numel() == 0:
            return torch.empty(query_group_ids.shape[0], 0, dtype=torch.bool, device=query_group_ids.device)
        groups = self.group_ids.to(query_group_ids.device)
        return query_group_ids.long().view(-1, 1) != groups.view(1, -1)


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positives = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits.log_softmax(dim=1)
    valid = positives.sum(dim=1) > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    losses = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid] / positives.sum(dim=1)[valid])
    return losses.mean()
