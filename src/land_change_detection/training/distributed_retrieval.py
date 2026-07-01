from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather

from land_change_detection.models.retrieval_heads import multi_positive_symmetric_info_nce


def distributed_multi_positive_info_nce(
    pair_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_to_pair: Tensor,
) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return multi_positive_symmetric_info_nce(pair_embeddings, text_embeddings, caption_to_pair)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pair_count = torch.tensor([pair_embeddings.shape[0]], device=pair_embeddings.device, dtype=torch.long)
    text_count = torch.tensor([text_embeddings.shape[0]], device=text_embeddings.device, dtype=torch.long)
    pair_counts = [torch.zeros_like(pair_count) for _ in range(world_size)]
    text_counts = [torch.zeros_like(text_count) for _ in range(world_size)]
    dist.all_gather(pair_counts, pair_count)
    dist.all_gather(text_counts, text_count)
    if len({int(value.item()) for value in pair_counts}) != 1:
        raise RuntimeError("Distributed retrieval requires equal pair batch sizes on every rank.")
    if len({int(value.item()) for value in text_counts}) != 1:
        raise RuntimeError("Distributed retrieval requires equal caption counts on every rank.")

    gathered_pairs = torch.cat(list(all_gather(pair_embeddings)), dim=0)
    gathered_texts = torch.cat(list(all_gather(text_embeddings)), dim=0)
    local_pair_count = pair_embeddings.shape[0]
    mapped = caption_to_pair + rank * local_pair_count
    mapping_parts = [torch.empty_like(mapped) for _ in range(world_size)]
    dist.all_gather(mapping_parts, mapped)
    global_mapping = torch.cat(mapping_parts, dim=0)
    return multi_positive_symmetric_info_nce(gathered_pairs, gathered_texts, global_mapping)
