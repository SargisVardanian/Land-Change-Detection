from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import json
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LogicalBatchContract:
    physical_micro_batch: int
    logical_physical_batch: int
    captions_per_pair: int

    @property
    def micro_batches_per_logical_batch(self) -> int:
        if self.logical_physical_batch % self.physical_micro_batch:
            raise ValueError("logical batch must be divisible by physical micro-batch")
        return self.logical_physical_batch // self.physical_micro_batch

    @property
    def logical_query_count(self) -> int:
        return self.logical_physical_batch * self.captions_per_pair

    @property
    def score_matrix_shape(self) -> tuple[int, int]:
        return self.logical_query_count, self.logical_physical_batch


class DeterministicRotatingCaptionCollator:
    """Equal-count caption sampling that covers every caption cyclically."""

    def __init__(self, captions_per_pair: int = 2, seed: int = 20260727, epoch: int = 0):
        if captions_per_pair < 1:
            raise ValueError("captions_per_pair must be positive")
        self.count = captions_per_pair
        self.seed = seed
        self.epoch = epoch

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        images = torch.stack([item.images for item in items])
        captions: list[str] = []
        normalized: list[str] = []
        query_pair_ids: list[str] = []
        for item in items:
            order = list(range(len(item.captions)))
            random.Random(f"{self.seed}:{item.pair_id}").shuffle(order)
            start = (self.epoch * self.count) % len(order)
            chosen = [order[(start + offset) % len(order)] for offset in range(self.count)]
            for index in chosen:
                captions.append(item.captions[index])
                normalized.append(item.normalized_captions[index])
                query_pair_ids.append(item.pair_id)
        return {
            "images": images,
            "pair_ids": [item.pair_id for item in items],
            "captions": captions,
            "normalized_captions": normalized,
            "query_pair_ids": query_pair_ids,
            "change_status": [item.change_status for item in items],
            "metadata": [item.metadata for item in items],
        }


def hard_aware_epoch_order(
    pair_ids: Sequence[str],
    *,
    logical_batch: int,
    seed: int,
    hard_by_pair: dict[str, Sequence[str]] | None = None,
) -> tuple[list[int], dict[str, int]]:
    """Build equal-weight epoch order while co-locating model-mined neighbours.

    Every physical item occurs once before deterministic circular padding. The
    padding offset rotates with the seed so the unavoidable remainder is not
    assigned permanently to the same items.
    """
    if logical_batch <= 1:
        raise ValueError("logical batch must be greater than one")
    if len(pair_ids) < logical_batch:
        raise ValueError("dataset smaller than logical batch")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair ids must be unique")
    rng = random.Random(seed)
    index = {pair_id: i for i, pair_id in enumerate(pair_ids)}
    remaining = set(pair_ids)
    seeds = list(pair_ids)
    rng.shuffle(seeds)
    order: list[str] = []
    hard_edges = 0
    while remaining:
        seed_pair = next(pair for pair in seeds if pair in remaining)
        group: list[str] = []
        queue: deque[str] = deque([seed_pair])
        while len(group) < logical_batch and remaining:
            while queue and len(group) < logical_batch:
                pair = queue.popleft()
                if pair not in remaining:
                    continue
                remaining.remove(pair)
                group.append(pair)
                for candidate in hard_by_pair.get(pair, ()) if hard_by_pair else ():
                    if candidate in remaining:
                        queue.append(candidate)
                        hard_edges += 1
            if len(group) < logical_batch and remaining:
                random_pair = next(pair for pair in seeds if pair in remaining)
                queue.append(random_pair)
        order.extend(group)
    unique_count = len(order)
    padding = (-unique_count) % logical_batch
    if padding:
        offset = seed % unique_count
        circular = order[offset:] + order[:offset]
        order.extend(circular[:padding])
    return [index[pair] for pair in order], {
        "unique_physical_items": unique_count,
        "logical_padding_items": padding,
        "hard_neighbour_insertions": hard_edges,
        "logical_batches": len(order) // logical_batch,
    }


def _rng_state(device: torch.device) -> tuple[Tensor, Tensor | None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(state: tuple[Tensor, Tensor | None], device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def exact_grad_cache_backward(
    *,
    micro_batches: Sequence[Any],
    encode: Callable[[Any], tuple[Tensor, Tensor]],
    loss_function: Callable[[Tensor, Tensor], tuple[Tensor, dict[str, Tensor]]],
    device: torch.device,
    autocast_factory: Callable[[], Any] | None = None,
) -> tuple[Tensor, dict[str, Tensor], tuple[int, int]]:
    """Backpropagate one exact logical contrastive loss via recomputation.

    The first pass caches only detached embeddings. A single common loss is
    differentiated over their concatenation. Each micro-batch is then replayed
    with its original RNG state and the cached embedding gradients.
    """
    if not micro_batches:
        raise ValueError("at least one micro-batch is required")
    context = autocast_factory or nullcontext
    cached_text: list[Tensor] = []
    cached_pair: list[Tensor] = []
    rng_states: list[tuple[Tensor, Tensor | None]] = []
    for batch in micro_batches:
        rng_states.append(_rng_state(device))
        with torch.no_grad(), context():
            text, pair = encode(batch)
        cached_text.append(text.detach())
        cached_pair.append(pair.detach())
    text_leaf = torch.cat(cached_text).requires_grad_(True)
    pair_leaf = torch.cat(cached_pair).requires_grad_(True)
    loss, stats = loss_function(text_leaf, pair_leaf)
    loss.backward()
    if text_leaf.grad is None or pair_leaf.grad is None:
        raise RuntimeError("logical loss did not produce embedding gradients")
    text_grad = text_leaf.grad.detach()
    pair_grad = pair_leaf.grad.detach()
    text_offset = pair_offset = 0
    for batch, state, cached_q, cached_p in zip(
        micro_batches, rng_states, cached_text, cached_pair, strict=True
    ):
        _restore_rng(state, device)
        with context():
            text, pair = encode(batch)
        q_count, p_count = cached_q.shape[0], cached_p.shape[0]
        torch.autograd.backward(
            (text, pair),
            (
                text_grad[text_offset : text_offset + q_count],
                pair_grad[pair_offset : pair_offset + p_count],
            ),
        )
        text_offset += q_count
        pair_offset += p_count
    return loss.detach(), {key: value.detach() for key, value in stats.items()}, (
        text_leaf.shape[0],
        pair_leaf.shape[0],
    )


def mine_hard_negative_rows(
    *,
    query_vectors: Tensor,
    pair_vectors: Tensor,
    query_pair_ids: Sequence[str],
    gallery_pair_ids: Sequence[str],
    query_captions: Sequence[str],
    excluded_pair_ids: Sequence[set[str]],
    top_k: int,
    chunk_size: int = 256,
) -> list[dict[str, Any]]:
    if query_vectors.shape[0] != len(query_pair_ids) or query_vectors.shape[0] != len(query_captions):
        raise ValueError("query metadata does not match vectors")
    if pair_vectors.shape[0] != len(gallery_pair_ids):
        raise ValueError("gallery metadata does not match vectors")
    if len(excluded_pair_ids) != len(query_pair_ids):
        raise ValueError("excluded metadata does not match queries")
    gallery_index = {pair_id: i for i, pair_id in enumerate(gallery_pair_ids)}
    rows: list[dict[str, Any]] = []
    pair_vectors = pair_vectors.float()
    for start in range(0, query_vectors.shape[0], chunk_size):
        scores = query_vectors[start : start + chunk_size].float() @ pair_vectors.T
        for local in range(scores.shape[0]):
            query_index = start + local
            true_pair = query_pair_ids[query_index]
            excluded = set(excluded_pair_ids[query_index]) | {true_pair}
            for pair_id in excluded:
                column = gallery_index.get(pair_id)
                if column is not None:
                    scores[local, column] = -torch.inf
            values, columns = scores[local].topk(top_k)
            rows.append({
                "query_index": query_index,
                "query": query_captions[query_index],
                "pair_id": true_pair,
                "hard_pair_ids": [gallery_pair_ids[i] for i in columns.tolist()],
                "hard_scores": [float(value) for value in values],
                "excluded_pair_count": len(excluded),
            })
    return rows


def aggregate_hard_pairs(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        seen = set(output.get(pair_id, ()))
        ordered = output.setdefault(pair_id, [])
        for candidate in row["hard_pair_ids"]:
            if candidate != pair_id and candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
    return output


def save_hard_negative_cache(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def filip_token_patch_scores(
    text_tokens: Tensor,
    patch_tokens: Tensor,
    content_mask: Tensor,
    *,
    top_k: int = 4,
) -> Tensor:
    """Generic FILIP-style score; inputs are already in a shared space."""
    if text_tokens.ndim != 3 or patch_tokens.ndim != 3:
        raise ValueError("tokens must be [Q,L,D] and [P,N,D]")
    if text_tokens.shape[-1] != patch_tokens.shape[-1]:
        raise ValueError("text and patch token dimensions must match")
    if content_mask.shape != text_tokens.shape[:2]:
        raise ValueError("content mask shape mismatch")
    if top_k < 1 or top_k > patch_tokens.shape[1]:
        raise ValueError("invalid patch top-k")
    text = F.normalize(text_tokens, dim=-1)
    patches = F.normalize(patch_tokens, dim=-1)
    similarity = torch.einsum("qld,pnd->qpln", text, patches)
    evidence = similarity.topk(top_k, dim=-1).values.mean(dim=-1)
    valid = content_mask.to(evidence.dtype)[:, None, :]
    return (evidence * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


class LocalAlignmentProjection(nn.Module):
    def __init__(
        self,
        text_dim: int = 512,
        visual_dim: int = 768,
        local_dim: int = 256,
    ):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, local_dim)
        self.visual_projection = nn.Linear(visual_dim, local_dim)

    def forward(
        self, text_tokens: Tensor, patch_tokens: Tensor, content_mask: Tensor
    ) -> Tensor:
        return filip_token_patch_scores(
            self.text_projection(text_tokens),
            self.visual_projection(patch_tokens),
            content_mask,
        )


class LoRALinear(nn.Module):
    """Low-rank residual around a frozen Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha if alpha is not None else rank) / rank

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.scale * self.lora_b(self.lora_a(inputs))


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for component in path.split("."):
        module = getattr(module, component)
    return module


def final_attention_block_prefixes(
    root: nn.Module,
    block_container_path: str,
    *,
    count: int,
) -> list[str]:
    container = _resolve_module(root, block_container_path)
    candidates = [index for index, block in enumerate(container) if hasattr(block, "attn") or hasattr(block, "self_attn")]
    if len(candidates) < count:
        raise ValueError("not enough attention blocks for requested LoRA scope")
    return [f"{block_container_path}.{index}" for index in candidates[-count:]]


def inject_lora_attention_projections(
    root: nn.Module,
    *,
    approved_block_prefixes: Sequence[str],
    projection_names: Sequence[str],
    rank: int = 8,
    alpha: float | None = None,
) -> list[str]:
    """Replace only explicitly approved attention projections with LoRA."""
    for parameter in root.parameters():
        parameter.requires_grad_(False)
    replaced: list[str] = []
    for block_prefix in approved_block_prefixes:
        block = _resolve_module(root, block_prefix)
        attention = getattr(block, "attn", getattr(block, "self_attn", None))
        if attention is None:
            raise ValueError(f"approved block has no attention module: {block_prefix}")
        attention_name = "attn" if hasattr(block, "attn") else "self_attn"
        for projection_name in projection_names:
            projection = getattr(attention, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise ValueError(f"missing Linear projection {block_prefix}.{attention_name}.{projection_name}")
            setattr(attention, projection_name, LoRALinear(projection, rank=rank, alpha=alpha))
            replaced.append(f"{block_prefix}.{attention_name}.{projection_name}")
    return replaced
