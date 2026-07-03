from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    classify_caption_semantics,
    semantic_teacher_relevance_matrix,
    stable_caption_group_ids,
)
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel


@dataclass(frozen=True)
class RetrievalCorpus:
    pair_embeddings: Tensor
    text_embeddings: Tensor
    caption_to_pair: Tensor
    caption_group_ids: Tensor
    pair_ids: list[str]
    captions: list[str]
    pair_mask_fractions: Tensor
    encode_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    dataset_names: list[str] | None = None
    teacher_text_embeddings: Tensor | None = None


@dataclass(frozen=True)
class RetrievalRankResult:
    similarities: Tensor
    ranked_candidate_indices: Tensor
    duplicate_aware_ranks: Tensor
    exact_pair_ranks: Tensor
    positive_mask: Tensor
    positive_counts: Tensor
    candidate_tie_keys: Tensor


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
    teacher_text_embeddings: list[Tensor] = []
    caption_to_pair_all: list[Tensor] = []
    caption_group_ids_all: list[Tensor] = []
    pair_mask_fractions: list[Tensor] = []
    pair_ids: list[str] = []
    dataset_names: list[str] = []
    captions: list[str] = []
    pair_offset = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch_pair_ids = [str(pair_id) for pair_id in batch["pair_ids"]]
            pair_ids.extend(batch_pair_ids)
            dataset_names.extend(str(name) for name in batch.get("dataset_names", ["unknown"] * len(batch_pair_ids)))
            captions.extend(str(caption) for caption in batch["captions"])
            raw_mask_fractions = batch.get("mask_fractions")
            if raw_mask_fractions is None:
                pair_mask_fractions.append(torch.full((len(batch_pair_ids),), float("nan")))
            else:
                pair_mask_fractions.append(torch.as_tensor(raw_mask_fractions, dtype=torch.float32).cpu())
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
            teacher_text_embeddings.append(output.teacher_text_embedding.float().cpu())
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
            pair_mask_fractions=torch.empty(0, dtype=torch.float32),
            encode_seconds=encode_seconds,
            peak_allocated_vram_bytes=peak_allocated,
            peak_reserved_vram_bytes=peak_reserved,
            dataset_names=[],
            teacher_text_embeddings=None,
        )

    return RetrievalCorpus(
        pair_embeddings=torch.cat(pair_embeddings),
        text_embeddings=torch.cat(text_embeddings),
        caption_to_pair=torch.cat(caption_to_pair_all),
        caption_group_ids=torch.cat(caption_group_ids_all),
        pair_ids=pair_ids,
        captions=captions,
        pair_mask_fractions=torch.cat(pair_mask_fractions),
        encode_seconds=encode_seconds,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
        dataset_names=dataset_names,
        teacher_text_embeddings=torch.cat(teacher_text_embeddings),
    )


def _rank_summary(ranks: Tensor, prefix: str = "") -> dict[str, float | int]:
    if ranks.numel() == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}empty": True,
            f"{prefix}R@1": 0.0,
            f"{prefix}R@5": 0.0,
            f"{prefix}R@10": 0.0,
            f"{prefix}MRR": 0.0,
            f"{prefix}median_rank": 0.0,
            f"{prefix}mean_rank": 0.0,
        }
    ranks = ranks.float()
    return {
        f"{prefix}count": int(ranks.numel()),
        f"{prefix}empty": False,
        f"{prefix}R@1": float((ranks <= 1).float().mean().item()),
        f"{prefix}R@5": float((ranks <= 5).float().mean().item()),
        f"{prefix}R@10": float((ranks <= 10).float().mean().item()),
        f"{prefix}MRR": float((1.0 / ranks).mean().item()),
        f"{prefix}median_rank": float(ranks.median().item()),
        f"{prefix}mean_rank": float(ranks.mean().item()),
    }


def _masked_rank_summary(ranks: Tensor, mask: Tensor, prefix: str) -> dict[str, float | int]:
    mask = mask.to(dtype=torch.bool)
    return _rank_summary(ranks[mask], prefix=prefix)


def _semantic_retrieval_summary(similarities: Tensor, relevance: Tensor, *, prefix: str = "") -> dict[str, float | int | bool]:
    if similarities.numel() == 0 or relevance.numel() == 0:
        return {
            f"{prefix}semantic_recall@1": 0.0,
            f"{prefix}semantic_recall@5": 0.0,
            f"{prefix}semantic_recall@10": 0.0,
            f"{prefix}semantic_nDCG@5": 0.0,
            f"{prefix}semantic_nDCG@10": 0.0,
            f"{prefix}semantic_query_count": 0,
            f"{prefix}semantic_empty": True,
        }
    positives = relevance > 0
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return {
            f"{prefix}semantic_recall@1": 0.0,
            f"{prefix}semantic_recall@5": 0.0,
            f"{prefix}semantic_recall@10": 0.0,
            f"{prefix}semantic_nDCG@5": 0.0,
            f"{prefix}semantic_nDCG@10": 0.0,
            f"{prefix}semantic_query_count": 0,
            f"{prefix}semantic_empty": True,
        }
    sims = similarities[valid]
    rel = relevance[valid].float()
    ranked = torch.argsort(sims, dim=1, descending=True, stable=True)
    gathered_positive = positives[valid].gather(1, ranked)
    result: dict[str, float | int | bool] = {
        f"{prefix}semantic_query_count": int(valid.sum().item()),
        f"{prefix}semantic_empty": False,
    }
    for k in (1, 5, 10):
        kk = min(k, gathered_positive.shape[1])
        result[f"{prefix}semantic_recall@{k}"] = float(gathered_positive[:, :kk].any(dim=1).float().mean().item())
    gains = rel.gather(1, ranked)
    discounts = 1.0 / torch.log2(torch.arange(rel.shape[1], dtype=torch.float32, device=rel.device) + 2.0)
    ideal = torch.sort(rel, dim=1, descending=True).values
    for k in (5, 10):
        kk = min(k, rel.shape[1])
        dcg = (gains[:, :kk] * discounts[:kk]).sum(dim=1)
        idcg = (ideal[:, :kk] * discounts[:kk]).sum(dim=1).clamp_min(1e-12)
        result[f"{prefix}semantic_nDCG@{k}"] = float((dcg / idcg).mean().item())
    return result


def _semantic_ranks(similarities: Tensor, relevance: Tensor) -> Tensor:
    positives = relevance > 0
    best_positive = similarities.masked_fill(~positives, float("-inf")).max(dim=1).values
    return (similarities > best_positive[:, None]).sum(dim=1).long() + 1


def _margin_summary(values: Tensor, prefix: str) -> dict[str, float | int]:
    if values.numel() == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}min": 0.0,
            f"{prefix}mean": 0.0,
            f"{prefix}p50": 0.0,
            f"{prefix}p10": 0.0,
        }
    values = values.float()
    return {
        f"{prefix}count": int(values.numel()),
        f"{prefix}min": float(values.min().item()),
        f"{prefix}mean": float(values.mean().item()),
        f"{prefix}p50": float(torch.quantile(values, 0.5).item()),
        f"{prefix}p10": float(torch.quantile(values, 0.1).item()),
    }


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _hash_tensor(tensor: Tensor) -> str:
    digest = hashlib.blake2b(digest_size=16)
    cpu = tensor.detach().cpu().contiguous()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _candidate_tie_keys(pair_ids: list[str], candidate_count: int) -> Tensor:
    if len(pair_ids) != candidate_count:
        return torch.arange(candidate_count, dtype=torch.long)
    keys: list[int] = []
    for pair_id in pair_ids:
        payload = str(pair_id).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        keys.append(value & 0x7FFF_FFFF_FFFF_FFFF)
    return torch.tensor(keys, dtype=torch.long)


def _validate_corpus(corpus: RetrievalCorpus) -> None:
    pair_count = int(corpus.pair_embeddings.shape[0])
    query_count = int(corpus.text_embeddings.shape[0])
    if corpus.pair_embeddings.ndim != 2 or corpus.text_embeddings.ndim != 2:
        raise ValueError("pair_embeddings and text_embeddings must be rank-2 tensors")
    if corpus.pair_embeddings.shape[1] != corpus.text_embeddings.shape[1]:
        raise ValueError("pair/text embedding dimensions do not match")
    if corpus.caption_to_pair.shape != (query_count,):
        raise ValueError("caption_to_pair must have one entry per text query")
    if corpus.caption_group_ids.shape != (query_count,):
        raise ValueError("caption_group_ids must have one entry per text query")
    if len(corpus.captions) != query_count:
        raise ValueError("captions length must match text query count")
    if len(corpus.pair_ids) != pair_count:
        raise ValueError("pair_ids length must match candidate pair count")
    if corpus.dataset_names is not None and len(corpus.dataset_names) != pair_count:
        raise ValueError("dataset_names length must match candidate pair count")
    if len(set(corpus.pair_ids)) != len(corpus.pair_ids):
        raise ValueError("pair_ids must be unique for deterministic retrieval ranking")
    if query_count:
        mapping = corpus.caption_to_pair.long()
        if int(mapping.min().item()) < 0 or int(mapping.max().item()) >= pair_count:
            raise ValueError("caption_to_pair contains an out-of-range candidate index")


def build_duplicate_aware_positive_mask(corpus: RetrievalCorpus) -> Tensor:
    _validate_corpus(corpus)
    query_count = int(corpus.text_embeddings.shape[0])
    pair_count = int(corpus.pair_embeddings.shape[0])
    mapping = corpus.caption_to_pair.long()
    groups = corpus.caption_group_ids.long()
    columns = torch.arange(query_count, dtype=torch.long)
    caption_pair_matrix = torch.zeros(query_count, pair_count, dtype=torch.bool)
    caption_pair_matrix[columns, mapping] = True
    same_group = groups[:, None] == groups[None, :]
    positives = (same_group.float() @ caption_pair_matrix.float()).to(torch.bool)
    if not positives[columns, mapping].all():
        raise RuntimeError("Positive mask lost an original caption-to-pair association")
    return positives


def similarity_matrix(
    corpus: RetrievalCorpus,
    *,
    query_chunk_size: int | None = None,
    candidate_chunk_size: int | None = None,
) -> Tensor:
    _validate_corpus(corpus)
    query_count = int(corpus.text_embeddings.shape[0])
    candidate_count = int(corpus.pair_embeddings.shape[0])
    if query_chunk_size is None or query_chunk_size <= 0:
        query_chunk_size = max(query_count, 1)
    if candidate_chunk_size is None or candidate_chunk_size <= 0:
        candidate_chunk_size = max(candidate_count, 1)
    output = torch.empty(query_count, candidate_count, dtype=torch.float32)
    text = corpus.text_embeddings.float()
    pairs = corpus.pair_embeddings.float()
    for query_start in range(0, query_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, query_count)
        for candidate_start in range(0, candidate_count, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, candidate_count)
            output[query_start:query_end, candidate_start:candidate_end] = (
                text[query_start:query_end] @ pairs[candidate_start:candidate_end].T
            )
    return output


def stable_ranked_candidate_indices(similarities: Tensor, candidate_tie_keys: Tensor) -> Tensor:
    if similarities.ndim != 2:
        raise ValueError("similarities must be a rank-2 matrix")
    if candidate_tie_keys.shape != (similarities.shape[1],):
        raise ValueError("candidate_tie_keys must have one key per candidate")
    key_order = torch.argsort(candidate_tie_keys.cpu(), stable=True)
    ranked_rows: list[Tensor] = []
    scores_cpu = similarities.detach().cpu()
    for row in scores_cpu:
        score_order = torch.argsort(row[key_order], descending=True, stable=True)
        ranked_rows.append(key_order[score_order])
    return torch.stack(ranked_rows, dim=0)


def compute_retrieval_ranks(
    corpus: RetrievalCorpus,
    *,
    query_chunk_size: int | None = None,
    candidate_chunk_size: int | None = None,
) -> RetrievalRankResult:
    similarities = similarity_matrix(
        corpus,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    positives = build_duplicate_aware_positive_mask(corpus)
    positive_counts = positives.sum(dim=1).long()
    if not torch.all(positive_counts > 0):
        raise ValueError("Every retrieval query must have at least one positive candidate")
    tie_keys = _candidate_tie_keys(corpus.pair_ids, similarities.shape[1])
    ranked = stable_ranked_candidate_indices(similarities, tie_keys)
    inverse_rank = torch.empty_like(ranked)
    inverse_rank.scatter_(1, ranked, torch.arange(ranked.shape[1], dtype=torch.long).view(1, -1).expand_as(ranked))
    duplicate_ranks = (inverse_rank.masked_fill(~positives, ranked.shape[1]).amin(dim=1) + 1).long()
    exact_ranks = (inverse_rank[torch.arange(ranked.shape[0]), corpus.caption_to_pair.long()] + 1).long()
    return RetrievalRankResult(
        similarities=similarities,
        ranked_candidate_indices=ranked,
        duplicate_aware_ranks=duplicate_ranks,
        exact_pair_ranks=exact_ranks,
        positive_mask=positives,
        positive_counts=positive_counts,
        candidate_tie_keys=tie_keys,
    )


def compute_retrieval_metrics(
    corpus: RetrievalCorpus,
    *,
    query_chunk_size: int | None = None,
    candidate_chunk_size: int | None = None,
) -> tuple[dict[str, float | int | bool | str], Tensor]:
    if corpus.pair_embeddings.numel() == 0:
        empty_metrics: dict[str, float | int | bool | str] = {
            "text_to_pair_R@1": 0.0,
            "text_to_pair_R@5": 0.0,
            "text_to_pair_R@10": 0.0,
            "MRR": 0.0,
            "median_rank": 0.0,
            "mean_rank": 0.0,
            "exact_pair_R@1": 0.0,
            "exact_pair_R@5": 0.0,
            "exact_pair_R@10": 0.0,
            "exact_pair_MRR": 0.0,
            "exact_pair_median_rank": 0.0,
            "exact_pair_mean_rank": 0.0,
            "pair_count": 0,
            "caption_count": 0,
            "num_queries": 0,
            "num_candidates": 0,
            "positive_count_min": 0,
            "positive_count_mean": 0.0,
            "positive_count_max": 0,
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
            "semantic_recall@1": 0.0,
            "semantic_recall@5": 0.0,
            "semantic_recall@10": 0.0,
            "semantic_nDCG@5": 0.0,
            "semantic_nDCG@10": 0.0,
        }
        return empty_metrics, torch.empty(0, 0)

    search_started = time.perf_counter()
    rank_result = compute_retrieval_ranks(
        corpus,
        query_chunk_size=query_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
    )
    search_seconds = time.perf_counter() - search_started

    similarities = rank_result.similarities
    ranks = rank_result.duplicate_aware_ranks.float()
    exact = rank_result.exact_pair_ranks.float()
    frequencies = rank_result.positive_counts
    ranked_scores = similarities.gather(1, rank_result.ranked_candidate_indices)
    if ranked_scores.shape[1] > 1:
        top1_top2_margin = ranked_scores[:, 0] - ranked_scores[:, 1]
    else:
        top1_top2_margin = torch.zeros(ranked_scores.shape[0], dtype=torch.float32)
    positive_scores = similarities.masked_fill(~rank_result.positive_mask, float("-inf"))
    negative_scores = similarities.masked_fill(rank_result.positive_mask, float("-inf"))
    best_positive_scores = positive_scores.max(dim=1).values
    best_negative_scores = negative_scores.max(dim=1).values
    best_positive_minus_best_negative = best_positive_scores - best_negative_scores
    best_positive_minus_best_negative = torch.where(
        torch.isfinite(best_positive_minus_best_negative),
        best_positive_minus_best_negative,
        torch.zeros_like(best_positive_minus_best_negative),
    )
    optimistic_ranks = (similarities > best_positive_scores[:, None]).sum(dim=1).long() + 1
    pessimistic_ranks = (similarities >= best_positive_scores[:, None]).sum(dim=1).long()
    exact_scores = similarities[torch.arange(similarities.shape[0]), corpus.caption_to_pair.long()]
    exact_tie_count = int(((similarities == exact_scores[:, None]).sum(dim=1) > 1).sum().item())
    near_tie_counts = {
        eps: int((top1_top2_margin <= eps).sum().item())
        for eps in (1e-6, 1e-5, 1e-4)
    }
    pair_count = int(corpus.pair_embeddings.shape[0])
    caption_count = int(corpus.text_embeddings.shape[0])

    metrics: dict[str, float | int | bool | str] = {
        "text_to_pair_R@1": float((ranks <= 1).float().mean().item()),
        "text_to_pair_R@5": float((ranks <= 5).float().mean().item()),
        "text_to_pair_R@10": float((ranks <= 10).float().mean().item()),
        "MRR": float((1.0 / ranks).mean().item()),
        "median_rank": float(ranks.median().item()),
        "mean_rank": float(ranks.mean().item()),
        "exact_pair_R@1": float((exact <= 1).float().mean().item()),
        "exact_pair_R@5": float((exact <= 5).float().mean().item()),
        "exact_pair_R@10": float((exact <= 10).float().mean().item()),
        "exact_pair_MRR": float((1.0 / exact).mean().item()),
        "exact_pair_median_rank": float(exact.median().item()),
        "exact_pair_mean_rank": float(exact.mean().item()),
        "exact_pair_ranks": [int(value) for value in exact.long().tolist()],
        "pair_count": pair_count,
        "caption_count": caption_count,
        "num_queries": caption_count,
        "num_candidates": pair_count,
        "positive_count_min": int(frequencies.min().item()),
        "positive_count_mean": float(frequencies.float().mean().item()),
        "positive_count_max": int(frequencies.max().item()),
        "exact_tie_count": exact_tie_count,
        "near_tie_count_eps_1e-6": near_tie_counts[1e-6],
        "near_tie_count_eps_1e-5": near_tie_counts[1e-5],
        "near_tie_count_eps_1e-4": near_tie_counts[1e-4],
        "tie_aware_optimistic_R@1": float((optimistic_ranks <= 1).float().mean().item()),
        "tie_aware_optimistic_R@5": float((optimistic_ranks <= 5).float().mean().item()),
        "tie_aware_optimistic_R@10": float((optimistic_ranks <= 10).float().mean().item()),
        "tie_aware_pessimistic_R@1": float((pessimistic_ranks <= 1).float().mean().item()),
        "tie_aware_pessimistic_R@5": float((pessimistic_ranks <= 5).float().mean().item()),
        "tie_aware_pessimistic_R@10": float((pessimistic_ranks <= 10).float().mean().item()),
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
        "pair_order_fingerprint": _hash_strings(corpus.pair_ids),
        "query_order_fingerprint": _hash_strings(
            [f"{int(pair)}\t{int(group)}\t{caption}" for pair, group, caption in zip(corpus.caption_to_pair.tolist(), corpus.caption_group_ids.tolist(), corpus.captions, strict=True)]
        ),
        "caption_to_pair_fingerprint": _hash_tensor(corpus.caption_to_pair.long()),
        "caption_group_fingerprint": _hash_tensor(corpus.caption_group_ids.long()),
        "rank_fingerprint": _hash_tensor(rank_result.duplicate_aware_ranks.long()),
        "exact_rank_fingerprint": _hash_tensor(rank_result.exact_pair_ranks.long()),
        "positive_mask_fingerprint": _hash_tensor(rank_result.positive_mask.to(torch.uint8)),
    }
    metrics.update(_margin_summary(top1_top2_margin, "top1_top2_margin_"))
    metrics.update(_margin_summary(best_positive_minus_best_negative, "best_positive_minus_best_negative_"))

    teacher_embeddings = corpus.teacher_text_embeddings if corpus.teacher_text_embeddings is not None else corpus.text_embeddings
    semantic_relevance = semantic_teacher_relevance_matrix(
        teacher_embeddings,
        corpus.captions,
        corpus.caption_to_pair.long(),
        corpus.caption_group_ids.long(),
        pair_count=pair_count,
        top_k=0,
    )
    semantic_ranks = _semantic_ranks(similarities, semantic_relevance).float()
    metrics.update(_semantic_retrieval_summary(similarities, semantic_relevance))

    if corpus.dataset_names is not None:
        pair_dataset_names = [str(name) for name in corpus.dataset_names]
        query_dataset_names = [pair_dataset_names[int(index)] for index in corpus.caption_to_pair.tolist()]
        metrics["dataset_order_fingerprint"] = _hash_strings(pair_dataset_names)
        macro_values: dict[str, list[float]] = {
            "semantic_recall@1": [],
            "semantic_recall@5": [],
            "semantic_recall@10": [],
            "semantic_nDCG@10": [],
        }
        for dataset_name in sorted(set(pair_dataset_names)):
            safe_name = dataset_name.replace("-", "_").replace(" ", "_")
            mask = torch.tensor([name == dataset_name for name in query_dataset_names], dtype=torch.bool)
            metrics.update(_masked_rank_summary(ranks, mask, f"{safe_name}_"))
            metrics.update(_masked_rank_summary(exact, mask, f"{safe_name}_exact_"))
            metrics.update(_masked_rank_summary(ranks, mask, f"{safe_name}_cross_"))
            metrics.update(_masked_rank_summary(semantic_ranks, mask, f"{safe_name}_cross_semantic_"))
            metrics.update(_masked_rank_summary(exact, mask, f"{safe_name}_cross_exact_"))
            cross_semantic = _semantic_retrieval_summary(similarities[mask], semantic_relevance[mask], prefix=f"{safe_name}_cross_")
            metrics.update(cross_semantic)
            for key in macro_values:
                prefixed = f"{safe_name}_cross_{key}"
                if prefixed in cross_semantic and not bool(cross_semantic.get(f"{safe_name}_cross_semantic_empty", False)):
                    macro_values[key].append(float(cross_semantic[prefixed]))
            metrics[f"{safe_name}_num_queries"] = int(mask.sum().item())
            metrics[f"{safe_name}_num_candidates"] = int(sum(name == dataset_name for name in pair_dataset_names))
            candidate_indices = [index for index, name in enumerate(pair_dataset_names) if name == dataset_name]
            query_indices = torch.nonzero(mask, as_tuple=False).flatten()
            if candidate_indices and query_indices.numel():
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
                candidate_set = {int(index) for index in candidate_indices}
                query_keep = [int(index) for index in query_indices.tolist() if int(corpus.caption_to_pair[int(index)].item()) in candidate_set]
                query_indices = torch.tensor(query_keep, dtype=torch.long)
            if candidate_indices and query_indices.numel():
                candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
                local_pairs = corpus.pair_embeddings[candidate_tensor]
                local_text = corpus.text_embeddings[query_indices]
                local_teacher = teacher_embeddings[query_indices]
                inverse_candidate = {global_index: local_index for local_index, global_index in enumerate(candidate_indices)}
                local_caption_to_pair = torch.tensor(
                    [inverse_candidate[int(corpus.caption_to_pair[int(query_index)].item())] for query_index in query_indices],
                    dtype=torch.long,
                )
                local_corpus = RetrievalCorpus(
                    pair_embeddings=local_pairs,
                    text_embeddings=local_text,
                    caption_to_pair=local_caption_to_pair,
                    caption_group_ids=corpus.caption_group_ids[query_indices],
                    pair_ids=[corpus.pair_ids[index] for index in candidate_indices],
                    captions=[corpus.captions[int(index)] for index in query_indices],
                    pair_mask_fractions=corpus.pair_mask_fractions[candidate_tensor] if corpus.pair_mask_fractions.numel() == len(pair_dataset_names) else torch.empty(0),
                    encode_seconds=0.0,
                    peak_allocated_vram_bytes=0,
                    peak_reserved_vram_bytes=0,
                    dataset_names=[dataset_name] * len(candidate_indices),
                    teacher_text_embeddings=local_teacher,
                )
                local_ranks = compute_retrieval_ranks(local_corpus)
                metrics.update(_rank_summary(local_ranks.duplicate_aware_ranks, f"{safe_name}_within_"))
                metrics.update(_rank_summary(local_ranks.exact_pair_ranks, f"{safe_name}_within_exact_"))
                local_similarities = similarity_matrix(local_corpus)
                local_relevance = semantic_teacher_relevance_matrix(
                    local_teacher,
                    local_corpus.captions,
                    local_corpus.caption_to_pair,
                    local_corpus.caption_group_ids,
                    pair_count=local_pairs.shape[0],
                    top_k=0,
                )
                metrics.update(_semantic_retrieval_summary(local_similarities, local_relevance, prefix=f"{safe_name}_within_"))
                metrics[f"{safe_name}_within_num_queries"] = int(query_indices.numel())
                metrics[f"{safe_name}_within_num_candidates"] = len(candidate_indices)
            else:
                metrics.update(_rank_summary(torch.empty(0, dtype=torch.long), f"{safe_name}_within_"))
                metrics.update(_rank_summary(torch.empty(0, dtype=torch.long), f"{safe_name}_within_exact_"))
                metrics.update(_semantic_retrieval_summary(torch.empty(0, 0), torch.empty(0, 0), prefix=f"{safe_name}_within_"))
        for key, values in macro_values.items():
            macro_key = "macro_" + key.replace("@", "_at_") if False else f"macro_{key}"
            metrics[macro_key] = float(sum(values) / len(values)) if values else 0.0
        metrics["macro_semantic_mean"] = float(
            (
                float(metrics.get("macro_semantic_recall@1", 0.0))
                + float(metrics.get("macro_semantic_recall@5", 0.0))
                + float(metrics.get("macro_semantic_recall@10", 0.0))
                + float(metrics.get("macro_semantic_nDCG@10", 0.0))
            )
            / 4.0
        )

    metrics.update(_masked_rank_summary(ranks, frequencies == 1, "unique_caption_"))
    metrics.update(_masked_rank_summary(ranks, (frequencies >= 2) & (frequencies <= 5), "rare_caption_"))
    metrics.update(_masked_rank_summary(ranks, frequencies > 5, "frequent_caption_"))
    metrics.update(_masked_rank_summary(exact, frequencies == 1, "unique_caption_exact_"))
    metrics.update(_masked_rank_summary(exact, (frequencies >= 2) & (frequencies <= 5), "rare_caption_exact_"))
    metrics.update(_masked_rank_summary(exact, frequencies > 5, "frequent_caption_exact_"))

    semantic = [classify_caption_semantics(caption) for caption in corpus.captions]
    for key in ("no_change", "changed", "appeared", "constructed", "added", "disappeared", "demolished", "removed", "increased", "expanded", "decreased", "reduced"):
        mask = torch.tensor([bool(item[key]) for item in semantic], dtype=torch.bool)
        metrics.update(_masked_rank_summary(ranks, mask, f"{key}_"))
        metrics.update(_masked_rank_summary(exact, mask, f"{key}_exact_"))
        metrics.update(_masked_rank_summary(semantic_ranks, mask, f"{key}_semantic_"))
    query_slices = {
        "detailed_query_": torch.tensor([bool(item["has_detail"]) for item in semantic], dtype=torch.bool),
        "directional_query_": torch.tensor([
            bool(item["appeared"] or item["constructed"] or item["added"] or item["disappeared"] or item["demolished"] or item["removed"] or item["increased"] or item["expanded"] or item["decreased"] or item["reduced"])
            for item in semantic
        ], dtype=torch.bool),
        "location_query_": torch.tensor([bool(item["has_location"]) for item in semantic], dtype=torch.bool),
        "count_query_": torch.tensor([bool(item["has_count"]) for item in semantic], dtype=torch.bool),
    }
    for prefix, mask in query_slices.items():
        summary = _masked_rank_summary(semantic_ranks, mask, prefix)
        metrics[f"{prefix}R@5"] = summary[f"{prefix}R@5"]
        metrics[f"{prefix}R@10"] = summary[f"{prefix}R@10"]
        metrics[f"{prefix}count"] = summary[f"{prefix}count"]

    if corpus.pair_mask_fractions.numel() == pair_count:
        query_mask_fraction = corpus.pair_mask_fractions[corpus.caption_to_pair]
        finite = torch.isfinite(query_mask_fraction)
        boundaries = getattr(corpus, "mask_fraction_boundaries", None) or (0.0, 0.01, 0.05)
        no_change_max, small_max, medium_max = [float(value) for value in boundaries]
        strata = {
            "mask_no_change_": finite & (query_mask_fraction <= no_change_max),
            "mask_small_change_": finite & (query_mask_fraction > no_change_max) & (query_mask_fraction <= small_max),
            "mask_medium_change_": finite & (query_mask_fraction > small_max) & (query_mask_fraction <= medium_max),
            "mask_large_change_": finite & (query_mask_fraction > medium_max),
        }
        for prefix, mask in strata.items():
            metrics.update(_masked_rank_summary(ranks, mask, prefix))
            metrics.update(_masked_rank_summary(exact, mask, f"{prefix}exact_"))

    return metrics, similarities


def relevance_aware_retrieval_metrics(
    model: UniChangeV2RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    config: base.RetrievalConfig,
) -> dict[str, float | int | bool]:
    corpus = collect_retrieval_corpus(model, loader, device, config)
    metrics, _ = compute_retrieval_metrics(
        corpus,
        query_chunk_size=int(getattr(config, "similarity_query_chunk_size", 0) or 0),
        candidate_chunk_size=int(getattr(config, "similarity_candidate_chunk_size", 0) or 0),
    )
    return metrics
