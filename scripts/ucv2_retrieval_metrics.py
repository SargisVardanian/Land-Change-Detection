from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.models.retrieval_heads import stable_caption_group_ids
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel


def relevance_aware_retrieval_metrics(
    model: UniChangeV2RetrievalModel,
    loader: DataLoader,
    device: torch.device,
    config: base.RetrievalConfig,
) -> dict[str, float | bool]:
    model.eval()
    pair_embeddings: list[Tensor] = []
    text_embeddings: list[Tensor] = []
    caption_to_pair_all: list[Tensor] = []
    caption_group_ids_all: list[Tensor] = []
    pair_offset = 0

    with torch.no_grad():
        for batch in loader:
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

    if not pair_embeddings:
        return {
            "text_to_pair_R@1": 0.0,
            "text_to_pair_R@5": 0.0,
            "text_to_pair_R@10": 0.0,
            "MRR": 0.0,
            "median_rank": 0.0,
            "exact_pair_R@1": 0.0,
            "duplicate_aware": True,
        }

    pairs = torch.cat(pair_embeddings)
    texts = torch.cat(text_embeddings)
    caption_to_pair = torch.cat(caption_to_pair_all)
    caption_group_ids = torch.cat(caption_group_ids_all)
    similarities = texts @ pairs.T

    group_to_pairs: dict[int, set[int]] = defaultdict(set)
    for group_id, pair_id in zip(caption_group_ids.tolist(), caption_to_pair.tolist(), strict=True):
        group_to_pairs[int(group_id)].add(int(pair_id))

    relevance_ranks: list[int] = []
    exact_ranks: list[int] = []
    for caption_index, exact_pair in enumerate(caption_to_pair.tolist()):
        order = torch.argsort(similarities[caption_index], descending=True)
        inverse_rank = torch.empty_like(order)
        inverse_rank[order] = torch.arange(order.numel())
        relevant_pairs = sorted(group_to_pairs[int(caption_group_ids[caption_index].item())])
        best_relevant_rank = min(int(inverse_rank[pair_id].item()) + 1 for pair_id in relevant_pairs)
        relevance_ranks.append(best_relevant_rank)
        exact_ranks.append(int(inverse_rank[int(exact_pair)].item()) + 1)

    ranks = torch.tensor(relevance_ranks, dtype=torch.float32)
    exact = torch.tensor(exact_ranks, dtype=torch.float32)
    return {
        "text_to_pair_R@1": float((ranks <= 1).float().mean().item()),
        "text_to_pair_R@5": float((ranks <= 5).float().mean().item()),
        "text_to_pair_R@10": float((ranks <= 10).float().mean().item()),
        "MRR": float((1.0 / ranks).mean().item()),
        "median_rank": float(ranks.median().item()),
        "exact_pair_R@1": float((exact <= 1).float().mean().item()),
        "duplicate_aware": True,
    }
