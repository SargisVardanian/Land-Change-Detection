from __future__ import annotations

import hashlib
import math
import re
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


_APPEARED_TERMS = {"appear", "appeared", "appears", "emerge", "emerged", "emerges", "new"}
_CONSTRUCTED_TERMS = {"build", "built", "construct", "constructed", "constructs", "developed", "created", "erected"}
_ADDED_TERMS = {"added", "addition", "installed"}
_DISAPPEARED_TERMS = {"disappear", "disappeared", "disappears", "gone", "lost", "vanished"}
_DEMOLISHED_TERMS = {"demolish", "demolished", "demolishes", "destroyed", "collapsed", "razed"}
_REMOVED_TERMS = {"removed", "cleared", "erased"}
_INCREASED_TERMS = {"increased", "increase", "grew", "grown", "growth", "more"}
_EXPANDED_TERMS = {"expand", "expanded", "expands", "expansion", "extended", "larger", "widened"}
_DECREASED_TERMS = {"decreased", "decrease", "declined", "less", "shrank", "shrunk"}
_REDUCED_TERMS = {"reduced", "reduction", "smaller", "contracted", "narrowed"}
_OBJECT_TERMS = {
    "building", "buildings", "house", "houses", "road", "roads", "water", "river", "lake",
    "field", "fields", "crop", "crops", "farmland", "forest", "trees", "vegetation",
    "urban", "construction", "bareland", "bare", "soil", "greenhouse", "greenhouses",
    "structure", "structures", "facility", "facilities", "settlement", "settlements",
    "parking", "lot", "bridge", "bridges",
}
_COUNT_TERMS = {"one", "two", "three", "four", "five", "several", "many", "multiple", "few", "single", "double"}
_LOCATION_TERMS = {
    "left", "right", "top", "bottom", "upper", "lower", "center", "central", "middle",
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "northern", "southern", "eastern", "western", "northeastern", "northwestern", "southeastern", "southwestern",
    "corner", "edge", "boundary", "near", "around", "beside",
}
_SIZE_TERMS = {"small", "large", "tiny", "major", "minor", "significant", "slight", "wide", "narrow", "dense", "sparse"}
_CHANGE_MODIFIERS = r"(?:any|significant|noticeable|visible|substantial|major|meaningful|temporal|notable|obvious|clear|material)"
_NO_CHANGE_PATTERNS = (
    re.compile(rf"\bno\s+(?:{_CHANGE_MODIFIERS}\s+){{0,3}}changes?\b"),
    re.compile(rf"\bwithout\s+(?:any\s+)?(?:{_CHANGE_MODIFIERS}\s+){{0,3}}changes?\b"),
    re.compile(r"\b(?:remained|remains|stayed|stays)\s+unchanged\b"),
    re.compile(r"\b(?:remained|remains|stayed|stays)\s+the\s+same\b"),
    re.compile(r"\bno\s+differences?\b"),
)
_NEGATION_TERMS = {"no", "not", "never", "without"}


def _is_negated(tokens: list[str], index: int, *, window: int = 4) -> bool:
    start = max(0, index - window)
    previous = tokens[start:index]
    if any(token in _NEGATION_TERMS for token in previous):
        return True
    return index >= 1 and tokens[index - 1].endswith("nt")


def _has_active_term(tokens: list[str], terms: set[str]) -> bool:
    return any(token in terms and not _is_negated(tokens, index) for index, token in enumerate(tokens))


def _has_negated_term(tokens: list[str], terms: set[str]) -> bool:
    return any(token in terms and _is_negated(tokens, index) for index, token in enumerate(tokens))


def _has_no_change_phrase(normalized: str) -> bool:
    return any(pattern.search(normalized) for pattern in _NO_CHANGE_PATTERNS)


def classify_caption_semantics(text: str) -> dict[str, Any]:
    """Central lightweight caption semantics for sampling, audit strata and filtering."""

    normalized = normalize_caption_text(text)
    token_list = normalized.split()
    tokens = set(token_list)
    has_digit_count = any(token.isdigit() for token in tokens)
    appeared = _has_active_term(token_list, _APPEARED_TERMS)
    constructed = _has_active_term(token_list, _CONSTRUCTED_TERMS)
    added = _has_active_term(token_list, _ADDED_TERMS)
    disappeared = _has_active_term(token_list, _DISAPPEARED_TERMS)
    demolished = _has_active_term(token_list, _DEMOLISHED_TERMS)
    removed = _has_active_term(token_list, _REMOVED_TERMS)
    increased = _has_active_term(token_list, _INCREASED_TERMS)
    expanded = _has_active_term(token_list, _EXPANDED_TERMS)
    decreased = _has_active_term(token_list, _DECREASED_TERMS)
    reduced = _has_active_term(token_list, _REDUCED_TERMS)
    positive_direction = appeared or constructed or added or increased or expanded
    negative_direction = disappeared or demolished or removed or decreased or reduced
    negated_direction = any(
        _has_negated_term(token_list, terms)
        for terms in (
            _APPEARED_TERMS,
            _CONSTRUCTED_TERMS,
            _ADDED_TERMS,
            _DISAPPEARED_TERMS,
            _DEMOLISHED_TERMS,
            _REMOVED_TERMS,
            _INCREASED_TERMS,
            _EXPANDED_TERMS,
            _DECREASED_TERMS,
            _REDUCED_TERMS,
        )
    )
    no_change = (
        _has_no_change_phrase(normalized)
        or "unchanged" in tokens
        or "without change" in normalized
        or "same" in tokens
        or (negated_direction and not (positive_direction or negative_direction))
    )
    changed = (
        not no_change
        and (
            "change" in tokens
            or "changed" in tokens
            or positive_direction
            or negative_direction
        )
    )
    object_terms = sorted(tokens & _OBJECT_TERMS)
    count_terms = sorted((tokens & _COUNT_TERMS) | {token for token in tokens if token.isdigit()})
    location_terms = sorted(tokens & _LOCATION_TERMS)
    size_terms = sorted(tokens & _SIZE_TERMS)
    return {
        "no_change": bool(no_change and not (positive_direction or negative_direction)),
        "changed": bool((changed or positive_direction or negative_direction) and not (no_change and not (positive_direction or negative_direction))),
        "appeared": appeared,
        "constructed": constructed,
        "added": added,
        "disappeared": disappeared,
        "demolished": demolished,
        "removed": removed,
        "increased": increased,
        "expanded": expanded,
        "decreased": decreased,
        "reduced": reduced,
        "object_terms": object_terms,
        "has_object": bool(object_terms),
        "count_terms": count_terms,
        "has_count": bool(count_terms or has_digit_count),
        "location_terms": location_terms,
        "has_location": bool(location_terms),
        "size_terms": size_terms,
        "has_size": bool(size_terms),
        "has_detail": bool(object_terms or count_terms or location_terms or size_terms),
    }


def semantic_direction_contradicts(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if query["no_change"] and candidate["changed"]:
        return True
    if query["changed"] and candidate["no_change"]:
        return True
    query_add = query["appeared"] or query["constructed"] or query["added"]
    cand_add = candidate["appeared"] or candidate["constructed"] or candidate["added"]
    query_remove = query["disappeared"] or query["demolished"] or query["removed"]
    cand_remove = candidate["disappeared"] or candidate["demolished"] or candidate["removed"]
    query_inc = query["increased"] or query["expanded"]
    cand_inc = candidate["increased"] or candidate["expanded"]
    query_dec = query["decreased"] or query["reduced"]
    cand_dec = candidate["decreased"] or candidate["reduced"]
    return bool((query_add and cand_remove) or (query_remove and cand_add) or (query_inc and cand_dec) or (query_dec and cand_inc))


def caption_detail_score(text: str) -> float:
    normalized = normalize_caption_text(text)
    tokens = normalized.split()
    semantics = classify_caption_semantics(text)
    length_score = min(len(tokens), 24) / 24.0
    direction_score = float(
        semantics["appeared"] or semantics["constructed"] or semantics["added"]
        or semantics["disappeared"] or semantics["demolished"] or semantics["removed"]
        or semantics["increased"] or semantics["expanded"] or semantics["decreased"] or semantics["reduced"]
    )
    return float(
        length_score
        + 0.8 * direction_score
        + 0.7 * float(semantics["has_object"])
        + 0.6 * float(semantics["has_count"])
        + 0.6 * float(semantics["has_location"])
        + 0.4 * float(semantics["has_size"])
    )


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


def semantic_teacher_relevance_matrix(
    teacher_text_embeddings: Tensor,
    captions: list[str],
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    pair_count: int,
    top_k: int = 8,
) -> Tensor:
    mapping = caption_to_pair.long()
    if teacher_text_embeddings.ndim != 2:
        raise ValueError("teacher_text_embeddings must be rank-2")
    if teacher_text_embeddings.shape[0] != mapping.numel() or len(captions) != mapping.numel():
        raise ValueError("teacher embeddings, captions and caption_to_pair must be aligned")
    if mapping.numel() == 0:
        raise ValueError("At least one caption is required")
    if int(mapping.min().item()) < 0 or int(mapping.max().item()) >= pair_count:
        raise ValueError("caption_to_pair contains an out-of-range pair index")
    device = teacher_text_embeddings.device
    teacher = F.normalize(teacher_text_embeddings.detach().float(), dim=-1)
    caption_sim = teacher @ teacher.T
    query_count = mapping.numel()
    relevance = torch.full((query_count, pair_count), float("-inf"), device=device)
    for pair_index in range(pair_count):
        members = torch.nonzero(mapping == pair_index, as_tuple=False).flatten()
        if members.numel():
            relevance[:, pair_index] = caption_sim[:, members].max(dim=1).values
    positives = build_caption_positive_mask(mapping, pair_count=pair_count, caption_group_ids=caption_group_ids).T
    relevance = relevance.masked_fill(positives.to(device), 1.0)

    semantics = [classify_caption_semantics(caption) for caption in captions]
    pair_semantics: list[list[dict[str, Any]]] = []
    for pair_index in range(pair_count):
        members = torch.nonzero(mapping == pair_index, as_tuple=False).flatten().tolist()
        pair_semantics.append([semantics[index] for index in members])
    for query_index, query_semantics in enumerate(semantics):
        for pair_index, candidate_semantics in enumerate(pair_semantics):
            if bool(positives[query_index, pair_index]):
                continue
            if any(semantic_direction_contradicts(query_semantics, candidate) for candidate in candidate_semantics):
                relevance[query_index, pair_index] = 0.0

    relevance = torch.clamp(relevance, min=0.0, max=1.0)
    if top_k > 0 and pair_count > top_k:
        keep = positives.to(device).clone()
        for query_index in range(query_count):
            remaining = max(int(top_k) - int(keep[query_index].sum().item()), 0)
            if remaining > 0:
                masked = relevance[query_index].masked_fill(keep[query_index], float("-inf"))
                keep[query_index, torch.topk(masked, k=min(remaining, pair_count)).indices] = True
        relevance = relevance.masked_fill(~keep, 0.0)
    relevance = relevance.masked_fill(positives.to(device), 1.0)
    return relevance.detach()


def semantic_soft_target_loss(
    logits_text_to_pair: Tensor,
    relevance: Tensor,
    *,
    teacher_temperature: float = 0.05,
) -> Tensor:
    if logits_text_to_pair.shape != relevance.shape:
        raise ValueError(f"logits and relevance must share shape, got {logits_text_to_pair.shape} and {relevance.shape}")
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    masked = relevance.float().masked_fill(relevance <= 0, float("-inf")) / teacher_temperature
    targets = masked.softmax(dim=1).detach()
    if not torch.isfinite(targets).all():
        raise ValueError("semantic teacher targets contain non-finite values")
    log_probs = logits_text_to_pair.float().log_softmax(dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


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


def semantic_text_to_pair_set_loss(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    teacher_text_embeddings: Tensor,
    captions: list[str],
    caption_to_pair: Tensor,
    caption_group_ids: Tensor,
    *,
    temperature: float = 0.07,
    logit_scale: Tensor | float | None = None,
    text_to_pair_weight: float = 0.75,
    pair_to_text_weight: float = 0.25,
    semantic_soft_target_weight: float = 0.25,
    semantic_teacher_top_k: int = 8,
    semantic_teacher_temperature: float = 0.05,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    if not 0.0 <= semantic_soft_target_weight <= 1.0:
        raise ValueError("semantic_soft_target_weight must be in [0, 1]")
    logits = _scaled_similarity_logits(pair_embeddings, text_embeddings, temperature=temperature, logit_scale=logit_scale)
    positives = build_caption_positive_mask(
        caption_to_pair,
        pair_count=pair_embeddings.shape[0],
        caption_group_ids=caption_group_ids,
    )
    set_text_to_pair = _positive_set_mass_loss(logits.T, positives.T)
    pair_to_text = _positive_set_mass_loss(logits, positives)
    relevance = semantic_teacher_relevance_matrix(
        teacher_text_embeddings,
        captions,
        caption_to_pair,
        caption_group_ids,
        pair_count=pair_embeddings.shape[0],
        top_k=semantic_teacher_top_k,
    ).to(logits.device)
    soft_loss = semantic_soft_target_loss(
        logits.T,
        relevance,
        teacher_temperature=semantic_teacher_temperature,
    )
    text_to_pair = (1.0 - semantic_soft_target_weight) * set_text_to_pair + semantic_soft_target_weight * soft_loss
    weight_sum = text_to_pair_weight + pair_to_text_weight
    if weight_sum <= 0.0:
        raise ValueError("at least one loss direction weight must be positive")
    loss = (text_to_pair_weight * text_to_pair + pair_to_text_weight * pair_to_text) / weight_sum
    if not return_diagnostics:
        return loss
    return loss, {
        "text_to_pair_loss": float(text_to_pair.detach().cpu()),
        "text_to_pair_set_mass_loss": float(set_text_to_pair.detach().cpu()),
        "semantic_soft_target_loss": float(soft_loss.detach().cpu()),
        "pair_to_text_loss": float(pair_to_text.detach().cpu()),
        "semantic_soft_target_weight": float(semantic_soft_target_weight),
        "semantic_teacher_top_k": int(semantic_teacher_top_k),
        "semantic_teacher_temperature": float(semantic_teacher_temperature),
        "teacher_positive_pairs": int((relevance > 0).sum().detach().cpu()),
        "teacher_target_min_positives": int((relevance > 0).sum(dim=1).min().detach().cpu()),
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
