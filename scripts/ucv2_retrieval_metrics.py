from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import stable_caption_group_ids
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel


@dataclass(frozen=True)
class RetrievalCorpus:
    pair_embeddings: Tensor
    text_embeddings: Tensor
    caption_to_pair: Tensor
    caption_group_ids: Tensor
    pair_ids: list[str]
    captions: list[str]
    encode_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_retrieval_corpus(
    model: UniChangeV2RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    config: base.RetrievalConfig,
) -> RetrievalCorpus:
    model.eval()
    pair_embeddings: list[Tensor] = []
    text_embeddings: list[Tensor] = []
    caption_to_pair_all: list[Tensor] = []
    caption_group_ids_all: list[Tensor] = []
    pair_ids: list[str] = []
    captions: list[str] = []
    pair_offset = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            pair_ids.extend(str(pair_id) for pair_id in batch["pair_ids"])
            captions.extend(str(caption) for caption in batch["captions"])
            batch = base._move_batch(batch, device)
            with base._amp_context(device, config.use_bf16):
                output = model(
                    batch["images"],
                    batch["captions"],
                    batch["caption_to_pair"],
                    batch["temporal_valid_mask"],
                )
            pair_embeddings.append(output.pair_embedding.float().cpu())
            text_embeddings.append(output.text_embedding.float().cpu())
            caption_to_pair_all.append(batch["caption_to_pair"].cpu() + pair_offset)
            caption_group_ids_all.append(stable_caption_group_ids(batch["captions"]).cpu())
            pair_offset += output.pair_embedding.shape[0]

    _synchronize(device)
    encode_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0

    if not pair_embeddings:
        empty = torch.empty(0, 0, dtype=torch.float32)
        return RetrievalCorpus(
            pair_embeddings=empty,
            text_embeddings=empty,
            caption_to_pair=torch.empty(0, dtype=torch.long),
            caption_group_ids=torch.empty(0, dtype=torch.long),
            pair_ids=[],
            captions=[],
            encode_seconds=encode_seconds,
            peak_allocated_vram_bytes=peak_allocated,
            peak_reserved_vram_bytes=peak_reserved,
        )

    return RetrievalCorpus(
        pair_embeddings=torch.cat(pair_embeddings),
        text_embeddings=torch.cat(text_embeddings),
        caption_to_pair=torch.cat(caption_to_pair_all),
        caption_group_ids=torch.cat(caption_group_ids_all),
        pair_ids=pair_ids,
        captions=captions,
        encode_seconds=encode_seconds,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
    )


def compute_retrieval_metrics(corpus: RetrievalCorpus) -> tuple[dict[str, float | int | bool], Tensor]:
    if corpus.pair_embeddings.numel() == 0:
        return (
            {
                "text_to_pair_R@1": 0.0,
                "text_to_pair_R@5": 0.0,
                "text_to_pair_R@10": 0.0,
                "MRR": 0.0,
                "median_rank": 0.0,
                "mean_rank": 0.0,
                "exact_pair_R@1": 0.0,
                "pair_count": 0,
                "caption_count": 0,
                "embedding_dim": 0,
                "encode_seconds": corpus.encode_seconds,
                "pairs_per_second": 0.0,
                "captions_per_second": 0.0,
                "similarity_search_seconds": 0.0,
                "search_queries_per_second": 0.0,
                "similarity_matrix_bytes": 0,
                "peak_allocated_vram_bytes": corpus.peak_allocated_vram_bytes,
                "peak_reserved_vram_bytes": corpus.peak_reserved_vram_bytes,
                "duplicate_aware": True,
            },
            torch.empty(0, 0),
        )

    search_started = time.perf_counter()
    similarities = corpus.text_embeddings @ corpus.pair_embeddings.T
    group_to_pairs: dict[int, set[int]] = defaultdict(set)
    for group_id, pair_id in zip(
        corpus.caption_group_ids.tolist(), corpus.caption_to_pair.tolist(), strict=True
    ):
        group_to_pairs[int(group_id)].add(int(pair_id))

    relevance_ranks: list[int] = []
    exact_ranks: list[int] = []
    for caption_index, exact_pair in enumerate(corpus.caption_to_pair.tolist()):
        order = torch.argsort(similarities[caption_index], descending=True)
        inverse_rank = torch.empty_like(order)
        inverse_rank[order] = torch.arange(order.numel())
        relevant_pairs = sorted(group_to_pairs[int(corpus.caption_group_ids[caption_index].item())])
        best_relevant_rank = min(int(inverse_rank[pair_id].item()) + 1 for pair_id in relevant_pairs)
        relevance_ranks.append(best_relevant_rank)
        exact_ranks.append(int(inverse_rank[int(exact_pair)].item()) + 1)
    search_seconds = time.perf_counter() - search_started

    ranks = torch.tensor(relevance_ranks, dtype=torch.float32)
    exact = torch.tensor(exact_ranks, dtype=torch.float32)
    pair_count = int(corpus.pair_embeddings.shape[0])
    caption_count = int(corpus.text_embeddings.shape[0])
    metrics: dict[str, float | int | bool] = {
        "text_to_pair_R@1": float((ranks <= 1).float().mean().item()),
        "text_to_pair_R@5": float((ranks <= 5).float().mean().item()),
        "text_to_pair_R@10": float((ranks <= 10).float().mean().item()),
        "MRR": float((1.0 / ranks).mean().item()),
        "median_rank": float(ranks.median().item()),
        "mean_rank": float(ranks.mean().item()),
        "exact_pair_R@1": float((exact <= 1).float().mean().item()),
        "pair_count": pair_count,
        "caption_count": caption_count,
        "embedding_dim": int(corpus.pair_embeddings.shape[1]),
        "encode_seconds": float(corpus.encode_seconds),
        "pairs_per_second": float(pair_count / max(corpus.encode_seconds, 1e-12)),
        "captions_per_second": float(caption_count / max(corpus.encode_seconds, 1e-12)),
        "similarity_search_seconds": float(search_seconds),
        "search_queries_per_second": float(caption_count / max(search_seconds, 1e-12)),
        "similarity_matrix_bytes": int(similarities.numel() * similarities.element_size()),
        "peak_allocated_vram_bytes": corpus.peak_allocated_vram_bytes,
        "peak_reserved_vram_bytes": corpus.peak_reserved_vram_bytes,
        "duplicate_aware": True,
    }
    return metrics, similarities


def relevance_aware_retrieval_metrics(
    model: UniChangeV2RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    config: base.RetrievalConfig,
) -> dict[str, float | int | bool]:
    corpus = collect_retrieval_corpus(model, loader, device, config)
    metrics, _ = compute_retrieval_metrics(corpus)
    return metrics
