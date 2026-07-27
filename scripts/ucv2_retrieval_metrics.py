from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import (
    classify_caption_semantics,
    semantic_teacher_relevance_matrix,
    stable_caption_group_ids,
)
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel
from land_change_detection.models.qcpr import QCPRPatchReranker, _structured_signature, structured_hard_negative_masks


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
    patch_tokens: Tensor | None = None
    mask_query_embeddings: Tensor | None = None
    qcpr_alpha: float = 1.0
    qcpr_beta: float = 0.0
    score_mode: str = "global"
    qcpr_architecture_version: str = "v1"
    pair_masks: Tensor | None = None
    changed_masks: Tensor | None = None
    segmentation_target_kinds: list[str] | None = None
    segmentation_weights: Tensor | None = None
    change_types: list[str | None] | None = None
    temporal_explanation_logits: Tensor | None = None
    text_token_embeddings: Tensor | None = None
    text_attention_mask: Tensor | None = None
    qcpr_reranker: QCPRPatchReranker | None = None


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
    text_token_embeddings: list[Tensor] = []
    text_attention_masks: list[Tensor] = []
    patch_tokens: list[Tensor] = []
    mask_query_embeddings: list[Tensor] = []
    caption_to_pair_all: list[Tensor] = []
    caption_group_ids_all: list[Tensor] = []
    pair_mask_fractions: list[Tensor] = []
    pair_masks: list[Tensor] = []
    changed_masks: list[Tensor] = []
    segmentation_target_kinds: list[str] = []
    segmentation_weights: list[Tensor] = []
    change_types: list[str | None] = []
    temporal_explanation_logits: list[Tensor] = []
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
            raw_masks = batch.get("masks")
            if raw_masks is not None:
                pair_masks.append(torch.as_tensor(raw_masks, dtype=torch.float32).cpu())
            raw_changed_masks = batch.get("changed_masks")
            if raw_changed_masks is not None:
                changed_masks.append(torch.as_tensor(raw_changed_masks, dtype=torch.float32).cpu())
            segmentation_target_kinds.extend(str(value) for value in batch.get("segmentation_target_kinds", ["none"] * len(batch_pair_ids)))
            raw_weights = batch.get("segmentation_weights")
            segmentation_weights.append(
                torch.as_tensor(raw_weights, dtype=torch.float32).cpu()
                if raw_weights is not None
                else torch.zeros(len(batch_pair_ids), dtype=torch.float32)
            )
            change_types.extend(batch.get("change_types", [None] * len(batch_pair_ids)))
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
            if output.patch_tokens is not None:
                patch_tokens.append(output.patch_tokens.float().cpu())
            if output.mask_query_embeddings is not None:
                mask_query_embeddings.append(output.mask_query_embeddings.float().cpu())
            if output.temporal_explanation_logits is not None:
                temporal_explanation_logits.append(output.temporal_explanation_logits.float().cpu())
            if output.text_token_embeddings is not None:
                text_token_embeddings.append(output.text_token_embeddings.float().cpu())
            if output.text_attention_mask is not None:
                text_attention_masks.append(output.text_attention_mask.bool().cpu())
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
            patch_tokens=None,
            mask_query_embeddings=None,
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
        patch_tokens=torch.cat(patch_tokens) if patch_tokens else None,
        mask_query_embeddings=torch.cat(mask_query_embeddings) if mask_query_embeddings else None,
        qcpr_alpha=float(getattr(getattr(model, "patch_reranker", None), "alpha", 1.0)),
        qcpr_beta=float(getattr(getattr(model, "patch_reranker", None), "beta", 0.0)),
        score_mode=("qcpr_v2" if patch_tokens and getattr(getattr(model, "patch_reranker", None), "architecture_version", "v1") == "v2" else "fused" if patch_tokens and mask_query_embeddings else "global"),
        qcpr_architecture_version=str(getattr(getattr(model, "patch_reranker", None), "architecture_version", "v1")),
        pair_masks=torch.cat(pair_masks) if pair_masks else None,
        changed_masks=torch.cat(changed_masks) if changed_masks else None,
        segmentation_target_kinds=segmentation_target_kinds,
        segmentation_weights=torch.cat(segmentation_weights) if segmentation_weights else None,
        change_types=[str(value) if value is not None else None for value in change_types],
        temporal_explanation_logits=torch.cat(temporal_explanation_logits) if temporal_explanation_logits else None,
        text_token_embeddings=(torch.cat([
            torch.nn.functional.pad(value, (0, 0, 0, max(item.shape[1] for item in text_token_embeddings) - value.shape[1]))
            for value in text_token_embeddings
        ]) if text_token_embeddings else None),
        text_attention_mask=(torch.cat([
            torch.nn.functional.pad(value, (0, max(item.shape[1] for item in text_attention_masks) - value.shape[1]))
            for value in text_attention_masks
        ]) if text_attention_masks else None),
        qcpr_reranker=getattr(model, "patch_reranker", None),
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


@torch.inference_mode()
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
    has_v1 = corpus.qcpr_architecture_version == "v1" and corpus.mask_query_embeddings is not None
    has_v2 = corpus.qcpr_architecture_version == "v2" and corpus.text_token_embeddings is not None and corpus.text_attention_mask is not None and corpus.qcpr_reranker is not None
    if corpus.patch_tokens is not None and has_v2:
        return retrieval_branch_similarity_matrices(
            corpus,
            query_chunk_size=query_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
        )["fused"]
    if corpus.patch_tokens is not None and (has_v1 or has_v2):
        patches = corpus.patch_tokens.float()
        local = torch.empty_like(output)
        token_patch = torch.empty_like(output) if has_v2 else None
        for query_start in range(0, query_count, query_chunk_size):
            query_end = min(query_start + query_chunk_size, query_count)
            for candidate_start in range(0, candidate_count, candidate_chunk_size):
                candidate_end = min(candidate_start + candidate_chunk_size, candidate_count)
                if has_v1:
                    queries = corpus.mask_query_embeddings.float()
                    logits = torch.einsum("qd,bnd->qbn", queries[query_start:query_end], patches[candidate_start:candidate_end])
                    local[query_start:query_end, candidate_start:candidate_end] = logits.sigmoid().amax(dim=-1)
                else:
                    reranker = corpus.qcpr_reranker
                    reranker_device = next(reranker.parameters()).device
                    descriptor = patches[candidate_start:candidate_end].to(reranker_device)
                    token = torch.nn.functional.normalize(reranker.token_projection(corpus.text_token_embeddings[query_start:query_end].to(reranker_device)), dim=-1)
                    attention = corpus.text_attention_mask[query_start:query_end].to(reranker_device)
                    token_similarity = torch.einsum("bnd,qld->qbnl", descriptor, token)
                    affinity = (token_similarity / reranker.logit_scale).masked_fill(~attention[:, None, None, :].bool(), -1e4)
                    attended = torch.einsum("qbnl,qld->qbnd", affinity.softmax(-1), token)
                    expanded = descriptor.unsqueeze(0).expand(query_end - query_start, -1, -1, -1)
                    logits = reranker.interaction_mlp(torch.cat((expanded, attended, expanded * attended), dim=-1)).squeeze(-1)
                    if corpus.temporal_explanation_logits is not None:
                        logits = logits + corpus.temporal_explanation_logits[candidate_start:candidate_end, :, 0].to(reranker_device).unsqueeze(0)
                    pooled = QCPRPatchReranker.masked_local_embedding(descriptor, logits)
                    local_chunk = torch.einsum("qd,qbd->qb", text[query_start:query_end].to(reranker_device), pooled)
                    local[query_start:query_end, candidate_start:candidate_end] = local_chunk.detach().cpu()
                    token_evidence = token_similarity.masked_fill(~attention[:, None, None, :].bool(), -1.0).amax(dim=-1)
                    top_k = max(1, min(4, token_evidence.shape[-1]))
                    token_patch[query_start:query_end, candidate_start:candidate_end] = (
                        token_evidence.topk(top_k, dim=-1).values.mean(dim=-1).detach().cpu()
                    )
                del logits
        if has_v2:
            reranker = corpus.qcpr_reranker
            raw = torch.stack((output, local, token_patch), dim=-1)
            calibrated = raw * reranker.branch_log_scales.detach().cpu().exp() + reranker.branch_biases.detach().cpu()
            return torch.einsum("qbk,k->qb", calibrated, reranker.fusion_logits.detach().cpu().softmax(dim=0))
        return corpus.qcpr_alpha * output + corpus.qcpr_beta * local
    return output


@torch.inference_mode()
def retrieval_branch_similarity_matrices(
    corpus: RetrievalCorpus,
    *,
    query_chunk_size: int | None = None,
    candidate_chunk_size: int | None = None,
    rerank_top_n: int = 0,
) -> dict[str, Tensor]:
    """Return independently inspectable global/local/token-patch/fused score matrices."""
    _validate_corpus(corpus)
    query_count = int(corpus.text_embeddings.shape[0])
    candidate_count = int(corpus.pair_embeddings.shape[0])
    query_chunk_size = query_chunk_size if query_chunk_size and query_chunk_size > 0 else max(query_count, 1)
    candidate_chunk_size = candidate_chunk_size if candidate_chunk_size and candidate_chunk_size > 0 else max(candidate_count, 1)
    global_scores = corpus.text_embeddings.float() @ corpus.pair_embeddings.float().T
    branches = {"global": global_scores}
    has_v1 = corpus.qcpr_architecture_version == "v1" and corpus.mask_query_embeddings is not None
    has_v2 = (
        corpus.qcpr_architecture_version == "v2"
        and corpus.text_token_embeddings is not None
        and corpus.text_attention_mask is not None
        and corpus.qcpr_reranker is not None
    )
    if corpus.patch_tokens is None or not (has_v1 or has_v2):
        branches["fused"] = global_scores
        if rerank_top_n > 0:
            branches["reranked"] = global_top_n_rerank_scores(global_scores, global_scores, rerank_top_n)
        return branches
    local_scores = torch.empty_like(global_scores)
    token_patch_scores = torch.empty_like(global_scores) if has_v2 else None
    for query_start in range(0, query_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, query_count)
        token = attention = query_global = None
        if has_v2:
            reranker = corpus.qcpr_reranker
            device = next(reranker.parameters()).device
            attention_cpu = corpus.text_attention_mask[query_start:query_end].bool()
            active_columns = attention_cpu.any(dim=0).nonzero(as_tuple=False).flatten()
            if active_columns.numel() == 0:
                token_start, token_end = 0, 1
            else:
                token_start, token_end = int(active_columns[0]), int(active_columns[-1]) + 1
            token_source = corpus.text_token_embeddings[query_start:query_end, token_start:token_end].to(device)
            token = F.normalize(reranker.token_projection(token_source), dim=-1)
            attention = attention_cpu[:, token_start:token_end].to(device)
            query_global = corpus.text_embeddings[query_start:query_end].to(device)
        for candidate_start in range(0, candidate_count, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, candidate_count)
            patches = corpus.patch_tokens[candidate_start:candidate_end].float()
            if has_v1:
                logits = torch.einsum(
                    "qd,bnd->qbn",
                    corpus.mask_query_embeddings[query_start:query_end].float(),
                    patches,
                )
                local_scores[query_start:query_end, candidate_start:candidate_end] = logits.sigmoid().amax(dim=-1)
            else:
                descriptor = patches.to(device)
                token_similarity = torch.einsum("bnd,qld->qbnl", descriptor, token)
                affinity = (token_similarity / reranker.logit_scale).masked_fill(~attention[:, None, None, :].bool(), -1e4)
                attended = torch.einsum("qbnl,qld->qbnd", affinity.softmax(-1), token)
                expanded = descriptor.unsqueeze(0).expand(query_end - query_start, -1, -1, -1)
                logits = reranker.interaction_mlp(torch.cat((expanded, attended, expanded * attended), dim=-1)).squeeze(-1)
                if corpus.temporal_explanation_logits is not None:
                    logits = logits + corpus.temporal_explanation_logits[candidate_start:candidate_end, :, 0].to(device).unsqueeze(0)
                pooled = QCPRPatchReranker.masked_local_embedding(descriptor, logits)
                local_scores[query_start:query_end, candidate_start:candidate_end] = torch.einsum(
                    "qd,qbd->qb", query_global, pooled
                ).detach().cpu()
                token_evidence = token_similarity.masked_fill(~attention[:, None, None, :].bool(), -1.0).amax(dim=-1)
                top_k = max(1, min(4, token_evidence.shape[-1]))
                token_patch_scores[query_start:query_end, candidate_start:candidate_end] = (
                    token_evidence.topk(top_k, dim=-1).values.mean(dim=-1).detach().cpu()
                )
    branches["local"] = local_scores
    if has_v2:
        reranker = corpus.qcpr_reranker
        branches["token_patch"] = token_patch_scores
        raw = torch.stack((global_scores, local_scores, token_patch_scores), dim=-1)
        calibrated = raw * reranker.branch_log_scales.detach().cpu().exp() + reranker.branch_biases.detach().cpu()
        branches["fused"] = torch.einsum(
            "qbk,k->qb", calibrated, reranker.fusion_logits.detach().cpu().softmax(dim=0)
        )
    else:
        branches["fused"] = corpus.qcpr_alpha * global_scores + corpus.qcpr_beta * local_scores
    if rerank_top_n > 0:
        branches["reranked"] = global_top_n_rerank_scores(global_scores, branches["fused"], rerank_top_n)
    return branches


def global_top_n_rerank_scores(global_scores: Tensor, rerank_scores: Tensor, top_n: int) -> Tensor:
    """Apply expensive local scores only inside the v1-compatible global candidate pool."""
    if global_scores.shape != rerank_scores.shape or global_scores.ndim != 2:
        raise ValueError("global_scores and rerank_scores must have equal [Q,B] shape")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    candidate_count = global_scores.shape[1]
    k = min(int(top_n), candidate_count)
    selected = global_scores.topk(k, dim=1).indices
    output = torch.full_like(rerank_scores, float("-inf"))
    output.scatter_(1, selected, rerank_scores.gather(1, selected))
    return output


def _safe_metric_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.casefold()).strip("_") or "unknown"


def _mask_metric_summary(predictions: Tensor, targets: Tensor) -> dict[str, float | int]:
    if predictions.numel() == 0:
        return {
            "count": 0,
            "Dice": 0.0,
            "IoU": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "predicted_mask_area_mean": 0.0,
            "target_mask_area_mean": 0.0,
        }
    predicted = predictions >= 0.5
    target = targets >= 0.5
    true_positive = (predicted & target).sum().float()
    false_positive = (predicted & ~target).sum().float()
    false_negative = (~predicted & target).sum().float()
    epsilon = 1e-8
    return {
        "count": int(predictions.shape[0]),
        "Dice": float((2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + epsilon)).item()),
        "IoU": float((true_positive / (true_positive + false_positive + false_negative + epsilon)).item()),
        "precision": float((true_positive / (true_positive + false_positive + epsilon)).item()),
        "recall": float((true_positive / (true_positive + false_negative + epsilon)).item()),
        "predicted_mask_area_mean": float(predicted.float().mean(dim=1).mean().item()),
        "target_mask_area_mean": float(target.float().mean(dim=1).mean().item()),
    }


def _mask_metric_summary_with_splits(predictions: Tensor, targets: Tensor) -> dict[str, float | int]:
    """Report aggregate, non-empty-target, and empty-target collapse diagnostics."""
    summary = _mask_metric_summary(predictions, targets)
    if predictions.numel() == 0:
        nonempty = empty = torch.zeros(0, dtype=torch.bool)
    else:
        nonempty = (targets >= 0.5).flatten(1).any(dim=1)
        empty = ~nonempty
    for split_name, keep in (("nonempty", nonempty), ("empty", empty)):
        split = _mask_metric_summary(predictions[keep], targets[keep])
        summary.update({f"{split_name}_{key}": value for key, value in split.items()})
    if torch.any(empty):
        predicted_empty = predictions[empty] >= 0.5
        summary["empty_target_false_positive_rate"] = float(predicted_empty.flatten(1).any(dim=1).float().mean().item())
        summary["empty_target_pixel_false_positive_rate"] = float(predicted_empty.float().mean().item())
    else:
        summary["empty_target_false_positive_rate"] = 0.0
        summary["empty_target_pixel_false_positive_rate"] = 0.0
    return summary


@torch.inference_mode()
def paired_candidate_mask_logits(
    corpus: RetrievalCorpus,
    query_indices: Tensor,
    candidate_indices: Tensor,
    *,
    pair_chunk_size: int = 16,
) -> Tensor:
    """Faithful aligned logits without allocating QxBxN or all paired interactions."""
    if query_indices.shape != candidate_indices.shape:
        raise ValueError("query_indices and candidate_indices must have identical shapes")
    if pair_chunk_size <= 0:
        pair_chunk_size = max(int(query_indices.numel()), 1)
    if query_indices.numel() == 0:
        patch_count = int(corpus.patch_tokens.shape[1])
        return torch.empty((0, patch_count), dtype=torch.float32)
    if corpus.qcpr_architecture_version == "v1":
        outputs = []
        for start in range(0, query_indices.numel(), pair_chunk_size):
            end = min(start + pair_chunk_size, query_indices.numel())
            patches = corpus.patch_tokens[candidate_indices[start:end]].float()
            queries = corpus.mask_query_embeddings[query_indices[start:end]].float()
            outputs.append(torch.einsum("qd,qnd->qn", queries, patches))
        return torch.cat(outputs, dim=0)
    reranker = corpus.qcpr_reranker
    device = next(reranker.parameters()).device
    outputs = []
    for start in range(0, query_indices.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, query_indices.numel())
        chunk_queries = query_indices[start:end]
        descriptor = corpus.patch_tokens[candidate_indices[start:end]].float().to(device)
        attention_cpu = corpus.text_attention_mask[chunk_queries].bool()
        active_columns = attention_cpu.any(dim=0).nonzero(as_tuple=False).flatten()
        token_start = int(active_columns[0]) if active_columns.numel() else 0
        token_end = int(active_columns[-1]) + 1 if active_columns.numel() else 1
        token = torch.nn.functional.normalize(
            reranker.token_projection(corpus.text_token_embeddings[chunk_queries, token_start:token_end].to(device)), dim=-1
        )
        attention = attention_cpu[:, token_start:token_end].to(device)
        affinity = torch.einsum("qnd,qld->qnl", descriptor, token) / reranker.logit_scale
        affinity = affinity.masked_fill(~attention[:, None, :].bool(), -1e4)
        attended = torch.einsum("qnl,qld->qnd", affinity.softmax(-1), token)
        logits = reranker.interaction_mlp(torch.cat((descriptor, attended, descriptor * attended), dim=-1))
        if corpus.temporal_explanation_logits is not None:
            logits = logits + corpus.temporal_explanation_logits[candidate_indices[start:end], :, 0].to(device).unsqueeze(-1)
        outputs.append(
            logits
            .squeeze(-1)
            .detach()
            .cpu()
        )
    return torch.cat(outputs, dim=0)


def supervised_mask_metrics(corpus: RetrievalCorpus) -> dict[str, float | int]:
    """Mask metrics for true supervised targets only; unsupervised rows never enter denominators."""
    empty = _mask_metric_summary_with_splits(torch.empty(0, 1), torch.empty(0, 1))
    metrics: dict[str, float | int] = {f"mask_{key}": value for key, value in empty.items()}
    metrics["predicted_mask_area_mean"] = 0.0
    metrics["target_mask_area_mean"] = 0.0
    if (
        corpus.patch_tokens is None
        or (corpus.mask_query_embeddings is None and corpus.qcpr_architecture_version != "v2")
        or corpus.pair_masks is None
        or corpus.segmentation_weights is None
        or corpus.segmentation_target_kinds is None
    ):
        return metrics
    pair_count = len(corpus.pair_ids)
    if (
        corpus.pair_masks.shape[0] != pair_count
        or corpus.segmentation_weights.shape != (pair_count,)
        or len(corpus.segmentation_target_kinds) != pair_count
    ):
        raise ValueError("Mask-supervision corpus fields must align with pair corpus")
    patch_count = int(corpus.patch_tokens.shape[1])
    side = int(round(patch_count**0.5))
    if side * side != patch_count:
        raise ValueError("QCPR patch grid must be square for mask metrics")
    mapping = corpus.caption_to_pair.long()
    supervised_pairs = corpus.segmentation_weights > 0
    keep = supervised_pairs[mapping]
    if not torch.any(keep):
        return metrics
    query_indices = torch.nonzero(keep, as_tuple=False).flatten()
    paired_indices = mapping[query_indices]
    logits = paired_candidate_mask_logits(corpus, query_indices, paired_indices)
    predictions = logits.sigmoid()
    targets = torch.nn.functional.interpolate(
        corpus.pair_masks[paired_indices, None].float(),
        size=(side, side),
        mode="nearest",
    )[:, 0].flatten(1)

    def add_summary(prefix: str, mask: Tensor) -> None:
        summary = _mask_metric_summary_with_splits(predictions[mask], targets[mask])
        metrics.update({f"{prefix}{key}": value for key, value in summary.items()})

    all_mask = torch.ones(query_indices.numel(), dtype=torch.bool)
    add_summary("mask_", all_mask)
    metrics["predicted_mask_area_mean"] = metrics["mask_predicted_mask_area_mean"]
    metrics["target_mask_area_mean"] = metrics["mask_target_mask_area_mean"]
    pair_dataset_names = corpus.dataset_names or ["unknown"] * pair_count
    query_dataset_names = [str(pair_dataset_names[int(index)]) for index in paired_indices.tolist()]
    query_kinds = [str(corpus.segmentation_target_kinds[int(index)]) for index in paired_indices.tolist()]
    query_change_types = [
        (corpus.change_types[int(index)] if corpus.change_types is not None else None)
        for index in paired_indices.tolist()
    ]
    for dataset in sorted(set(query_dataset_names)):
        add_summary(f"mask_dataset_{_safe_metric_name(dataset)}_", torch.tensor([value == dataset for value in query_dataset_names]))
    for kind in sorted(set(query_kinds)):
        add_summary(f"mask_target_kind_{_safe_metric_name(kind)}_", torch.tensor([value == kind for value in query_kinds]))
    for direction in ("appeared", "disappeared"):
        add_summary(
            f"mask_change_type_{direction}_",
            torch.tensor([value == direction for value in query_change_types]),
        )
    return metrics


def temporal_channel_mask_metrics(corpus: RetrievalCorpus) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if (
        corpus.temporal_explanation_logits is None
        or corpus.pair_masks is None
        or corpus.segmentation_weights is None
        or corpus.change_types is None
    ):
        for name in ("changed", "appeared", "disappeared"):
            metrics.update({f"mask_temporal_{name}_{key}": value for key, value in _mask_metric_summary_with_splits(torch.empty(0, 1), torch.empty(0, 1)).items()})
        return metrics
    logits = corpus.temporal_explanation_logits.float()
    patch_count = logits.shape[1]
    side = int(round(patch_count**0.5))
    if side * side != patch_count:
        raise ValueError("Temporal channel patch grid must be square")
    pair_targets = F.interpolate(corpus.pair_masks[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1)
    changed_source = corpus.changed_masks if corpus.changed_masks is not None else corpus.pair_masks
    changed_targets = F.interpolate(changed_source[:, None].float(), size=(side, side), mode="nearest")[:, 0].flatten(1)
    supervised = corpus.segmentation_weights > 0
    probabilities = logits.sigmoid().permute(0, 2, 1)
    specifications = {
        "changed": (0, supervised, changed_targets),
        "appeared": (1, supervised & torch.tensor([value == "appeared" for value in corpus.change_types]), pair_targets),
        "disappeared": (2, supervised & torch.tensor([value == "disappeared" for value in corpus.change_types]), pair_targets),
    }
    for name, (channel, keep, targets) in specifications.items():
        summary = _mask_metric_summary_with_splits(probabilities[keep, channel], targets[keep])
        metrics.update({f"mask_temporal_{name}_{key}": value for key, value in summary.items()})
    return metrics


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
    similarities: Tensor | None = None,
) -> RetrievalRankResult:
    if similarities is None:
        similarities = similarity_matrix(
            corpus,
            query_chunk_size=query_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
        )
    expected_shape = (int(corpus.text_embeddings.shape[0]), int(corpus.pair_embeddings.shape[0]))
    if similarities.shape != expected_shape:
        raise ValueError(f"similarities must have shape {expected_shape}, got {tuple(similarities.shape)}")
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


def _score_distribution(values: Tensor, prefix: str) -> dict[str, float | int]:
    finite = values.float()[torch.isfinite(values)]
    if finite.numel() == 0:
        return {f"{prefix}count": 0, f"{prefix}mean": 0.0, f"{prefix}p10": 0.0, f"{prefix}p50": 0.0, f"{prefix}p90": 0.0}
    return {
        f"{prefix}count": int(finite.numel()),
        f"{prefix}mean": float(finite.mean().item()),
        f"{prefix}p10": float(torch.quantile(finite, 0.1).item()),
        f"{prefix}p50": float(torch.quantile(finite, 0.5).item()),
        f"{prefix}p90": float(torch.quantile(finite, 0.9).item()),
    }


def _calibration_metrics(scores: Tensor, relevance: Tensor, bins: int = 10) -> dict[str, float]:
    probabilities = scores.float().sigmoid().flatten()
    labels = relevance.bool().float().flatten()
    brier = (probabilities - labels).square().mean()
    ece = probabilities.new_zeros(())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        keep = (probabilities >= lower) & (probabilities < upper if index < bins - 1 else probabilities <= upper)
        if torch.any(keep):
            ece = ece + keep.float().mean() * (probabilities[keep].mean() - labels[keep].mean()).abs()
    return {"ECE": float(ece.item()), "Brier": float(brier.item())}


def _structured_attribute_matches(
    corpus: RetrievalCorpus,
    ranked: Tensor,
) -> dict[str, float | int]:
    candidate_captions = [""] * len(corpus.pair_ids)
    for caption, pair_index in zip(corpus.captions, corpus.caption_to_pair.tolist(), strict=True):
        if not candidate_captions[int(pair_index)]:
            candidate_captions[int(pair_index)] = caption
    query_signatures = [_structured_signature(caption) for caption in corpus.captions]
    candidate_signatures = [_structured_signature(caption) for caption in candidate_captions]
    attribute_keys = {
        "object": "objects",
        "direction": "direction",
        "location": "locations",
        "count": "counts",
        "relation": "relation",
    }
    output: dict[str, float | int] = {}
    for label, key in attribute_keys.items():
        eligible = []
        matches = {1: [], 5: [], 10: []}
        for query_index, query in enumerate(query_signatures):
            query_value = query[key]
            present = bool(query_value) and query_value not in {"unknown", "none"}
            if not present:
                continue
            eligible.append(query_index)
            for k in matches:
                found = False
                for candidate_index in ranked[query_index, : min(k, ranked.shape[1])].tolist():
                    candidate_value = candidate_signatures[int(candidate_index)][key]
                    if isinstance(query_value, frozenset):
                        found = bool(query_value & candidate_value)
                    else:
                        found = query_value == candidate_value
                    if found:
                        break
                matches[k].append(found)
        output[f"{label}_match_count"] = len(eligible)
        for k, values in matches.items():
            output[f"{label}_match_R@{k}"] = float(sum(values) / len(values)) if values else 0.0
    return output


def retrieval_branch_diagnostics(
    corpus: RetrievalCorpus,
    branch_scores: dict[str, Tensor],
) -> dict[str, dict[str, float | int | bool]]:
    duplicate_positive = build_duplicate_aware_positive_mask(corpus)
    teacher = corpus.teacher_text_embeddings if corpus.teacher_text_embeddings is not None else corpus.text_embeddings
    semantic_relevance = semantic_teacher_relevance_matrix(
        teacher,
        corpus.captions,
        corpus.caption_to_pair.long(),
        corpus.caption_group_ids.long(),
        pair_count=len(corpus.pair_ids),
        top_k=0,
    )
    hard_masks = structured_hard_negative_masks(corpus.captions, corpus.caption_to_pair, len(corpus.pair_ids))
    hard_negative = torch.stack(tuple(hard_masks.values())).any(dim=0)
    tie_keys = _candidate_tie_keys(corpus.pair_ids, len(corpus.pair_ids))
    random_negative_indices = []
    for query_index in range(len(corpus.captions)):
        candidates = tie_keys.argsort(stable=True)
        choice = next((int(index) for index in candidates.tolist() if not duplicate_positive[query_index, index]), -1)
        random_negative_indices.append(choice)
    output: dict[str, dict[str, float | int | bool]] = {}
    for branch, scores in branch_scores.items():
        ranked = stable_ranked_candidate_indices(scores, tie_keys)
        inverse = torch.empty_like(ranked)
        inverse.scatter_(1, ranked, torch.arange(ranked.shape[1]).view(1, -1).expand_as(ranked))
        duplicate_ranks = inverse.masked_fill(~duplicate_positive, ranked.shape[1]).amin(dim=1) + 1
        exact_ranks = inverse[torch.arange(ranked.shape[0]), corpus.caption_to_pair.long()] + 1
        positive_scores = scores[duplicate_positive]
        random_values = [scores[index, candidate] for index, candidate in enumerate(random_negative_indices) if candidate >= 0]
        random_scores = torch.stack(random_values) if random_values else scores.new_empty(0)
        hard_scores = scores[hard_negative]
        best_positive = scores.masked_fill(~duplicate_positive, float("-inf")).max(dim=1).values
        best_negative = scores.masked_fill(duplicate_positive, float("-inf")).max(dim=1).values
        ranked_scores = scores.gather(1, ranked)
        metrics: dict[str, float | int | bool] = {}
        metrics.update(_semantic_retrieval_summary(scores, semantic_relevance))
        metrics.update(_rank_summary(duplicate_ranks, "duplicate_aware_"))
        metrics.update(_rank_summary(exact_ranks, "exact_pair_"))
        metrics.update(_score_distribution(positive_scores, "positive_score_"))
        metrics.update(_score_distribution(random_scores, "random_negative_score_"))
        metrics.update(_score_distribution(hard_scores, "hard_negative_score_"))
        metrics.update(_margin_summary(best_positive - best_negative, "best_positive_minus_best_negative_"))
        metrics.update(_margin_summary(ranked_scores[:, 0] - ranked_scores[:, 1], "top1_minus_top2_"))
        metrics.update(_calibration_metrics(scores, duplicate_positive))
        metrics.update(_structured_attribute_matches(corpus, ranked))
        output[branch] = metrics
    return output


def compute_retrieval_metrics(
    corpus: RetrievalCorpus,
    *,
    query_chunk_size: int | None = None,
    candidate_chunk_size: int | None = None,
    rank_result: RetrievalRankResult | None = None,
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
    if rank_result is None:
        rank_result = compute_retrieval_ranks(
            corpus, query_chunk_size=query_chunk_size, candidate_chunk_size=candidate_chunk_size,
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

    metrics.update(supervised_mask_metrics(corpus))
    metrics.update(temporal_channel_mask_metrics(corpus))

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
