from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def _rank_statistics(ranks: Tensor) -> dict[str, float]:
    values = ranks.float()
    result = {
        "mrr_full": float((1.0 / values).mean()),
        "mean_rank": float(values.mean()),
        "median_rank": float(values.median()),
    }
    for k in (1, 5, 10, 50, 100):
        result[f"candidate_hit_at_{k}"] = float((ranks <= k).float().mean())
    return result


def paired_clustered_rank_bootstrap(
    left_ranks: Tensor,
    right_ranks: Tensor,
    cluster_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap paired retrieval deltas without re-sorting a gallery.

    Physical-pair cluster IDs are sampled with replacement.  All caption
    queries belonging to each sampled physical pair are retained together.
    ``left_ranks`` and ``right_ranks`` must have been computed from the same
    query ordering and common gallery.
    """

    if left_ranks.ndim != 1 or right_ranks.shape != left_ranks.shape:
        raise ValueError("paired ranks must be equal one-dimensional tensors")
    if len(cluster_ids) != left_ranks.numel():
        raise ValueError("cluster IDs must align with ranks")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        groups[str(cluster_id)].append(index)
    if not groups:
        raise ValueError("bootstrap requires at least one physical-pair cluster")
    group_indices = [torch.tensor(indices, dtype=torch.long) for indices in groups.values()]
    generator = torch.Generator().manual_seed(seed)
    metric_names = tuple(_rank_statistics(left_ranks).keys())
    deltas: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled_groups = torch.randint(
            len(group_indices), (len(group_indices),), generator=generator
        )
        query_indices = torch.cat(
            [group_indices[index] for index in sampled_groups.tolist()]
        )
        left = _rank_statistics(left_ranks[query_indices])
        right = _rank_statistics(right_ranks[query_indices])
        for name in metric_names:
            deltas[name].append(left[name] - right[name])
    return {
        "unit": "physical_pair",
        "physical_pair_count": len(group_indices),
        "replicates": replicates,
        "seed": seed,
        "rankings_recomputed_per_replicate": False,
        "intervals_95": {
            name: {
                "lower": float(torch.quantile(torch.tensor(values), 0.025)),
                "median": float(torch.quantile(torch.tensor(values), 0.5)),
                "upper": float(torch.quantile(torch.tensor(values), 0.975)),
            }
            for name, values in deltas.items()
        },
    }


__all__ = ["paired_clustered_rank_bootstrap"]
