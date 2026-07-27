from __future__ import annotations

import torch
from torch import Tensor


def _ranks_from_scores(scores: Tensor, caption_to_pair: Tensor) -> list[int]:
    ranks: list[int] = []
    order = scores.argsort(dim=1, descending=True)
    for row_index, target in enumerate(caption_to_pair.tolist()):
        matches = (order[row_index] == int(target)).nonzero(as_tuple=False)
        ranks.append(int(matches[0, 0].item()) + 1 if matches.numel() else scores.shape[1] + 1)
    return ranks


def retrieval_metrics(pair_embeddings: Tensor, text_embeddings: Tensor, caption_to_pair: Tensor) -> dict[str, float]:
    scores = text_embeddings @ pair_embeddings.transpose(0, 1)
    ranks = _ranks_from_scores(scores, caption_to_pair)
    if not ranks:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "R@1": 0.0,
            "R@5": 0.0,
            "R@10": 0.0,
            "mrr": 0.0,
            "median_rank": 0.0,
            "map": 0.0,
        }
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    r1 = float((ranks_tensor <= 1).float().mean().item())
    r5 = float((ranks_tensor <= 5).float().mean().item())
    r10 = float((ranks_tensor <= 10).float().mean().item())
    mrr = float((1.0 / ranks_tensor).mean().item())
    return {
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "mrr": mrr,
        "median_rank": float(ranks_tensor.median().item()),
        "map": mrr,
    }


def dataset_retrieval_metrics(pair_embeddings: Tensor, text_embeddings: Tensor, caption_to_pair: Tensor) -> dict[str, float]:
    text_to_pair = retrieval_metrics(pair_embeddings, text_embeddings, caption_to_pair)
    scores = pair_embeddings @ text_embeddings.transpose(0, 1)
    pair_ranks: list[int] = []
    for pair_index in range(pair_embeddings.shape[0]):
        positives = (caption_to_pair == pair_index).nonzero(as_tuple=False).flatten()
        order = scores[pair_index].argsort(descending=True)
        positive_positions = [(order == positive).nonzero(as_tuple=False)[0, 0].item() + 1 for positive in positives]
        pair_ranks.append(min(positive_positions) if positive_positions else scores.shape[1] + 1)
    ranks = torch.tensor(pair_ranks, dtype=torch.float32)
    text_to_pair.update(
        {
            "pair_to_text_R@1": float((ranks <= 1).float().mean().item()),
            "pair_to_text_R@5": float((ranks <= 5).float().mean().item()),
            "pair_to_text_R@10": float((ranks <= 10).float().mean().item()),
        }
    )
    return text_to_pair


def write_full_rankings(path: str, pair_ids: list[str], captions: list[str], scores: Tensor) -> None:
    import json
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        order = scores.argsort(dim=1, descending=True)
        for caption_index, caption in enumerate(captions):
            ranked = [
                {"rank": rank + 1, "pair_id": pair_ids[int(pair_index)], "score": float(scores[caption_index, pair_index].item())}
                for rank, pair_index in enumerate(order[caption_index].tolist())
            ]
            handle.write(json.dumps({"caption_index": caption_index, "caption": caption, "rankings": ranked}) + "\n")
