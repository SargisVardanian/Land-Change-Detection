from __future__ import annotations

import pytest
import torch

from qcpr_siglip2.evaluation.bootstrap import paired_clustered_rank_bootstrap
from qcpr_siglip2.evaluation.retrieval import (
    candidate_hit_at_k,
    first_positive_ranks,
    full_gallery_metrics,
    map_at_k,
    mean_rank,
    median_rank,
    mrr_at_k,
    mrr_full,
    multi_positive_recall_at_k,
    precision_at_k,
)


def _scores_and_relevance() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.tensor(
        [
            [0.9, 0.2, 0.1, 0.0],
            [0.1, 0.7, 0.6, 0.2],
            [0.1, 0.2, 0.8, 0.7],
            [0.8, 0.6, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.9],
            [0.2, 0.8, 0.1, 0.7],
        ]
    )
    right = torch.tensor(
        [
            [0.5, 0.8, 0.2, 0.1],
            [0.2, 0.6, 0.9, 0.1],
            [0.7, 0.6, 0.5, 0.4],
            [0.9, 0.3, 0.2, 0.1],
            [0.8, 0.2, 0.1, 0.7],
            [0.3, 0.7, 0.2, 0.6],
        ]
    )
    relevance = torch.tensor(
        [
            [True, False, False, False],
            [False, True, True, False],
            [False, False, True, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    return left, right, relevance


def test_fused_full_gallery_metrics_match_independent_formulas() -> None:
    scores, _right, relevance = _scores_and_relevance()
    fused = full_gallery_metrics(scores, relevance, ks=(1, 2, 4))
    expected = {
        "mrr_full": mrr_full(scores, relevance),
        "mean_rank": mean_rank(scores, relevance),
        "median_rank": median_rank(scores, relevance),
    }
    for k in (1, 2, 4):
        expected[f"candidate_hit_at_{k}"] = candidate_hit_at_k(scores, relevance, k)
        expected[f"multi_positive_recall_at_{k}"] = multi_positive_recall_at_k(
            scores, relevance, k
        )
        expected[f"precision_at_{k}"] = precision_at_k(scores, relevance, k)
        expected[f"mrr_at_{k}"] = mrr_at_k(scores, relevance, k)
        expected[f"map_at_{k}"] = map_at_k(scores, relevance, k)
    assert fused == pytest.approx(expected)


def test_precomputed_rank_bootstrap_matches_gallery_reranking_reference() -> None:
    left, right, relevance = _scores_and_relevance()
    cluster_ids = ["a", "a", "b", "c", "c", "d"]
    seed = 23
    replicates = 31
    optimized = paired_clustered_rank_bootstrap(
        first_positive_ranks(left, relevance),
        first_positive_ranks(right, relevance),
        cluster_ids,
        seed=seed,
        replicates=replicates,
    )

    groups = [[0, 1], [2], [3, 4], [5]]
    generator = torch.Generator().manual_seed(seed)
    names = tuple(optimized["intervals_95"])
    deltas: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        sampled = torch.randint(len(groups), (len(groups),), generator=generator)
        indices = torch.tensor(
            [query for group in sampled.tolist() for query in groups[group]]
        )
        left_metrics = full_gallery_metrics(left[indices], relevance[indices])
        right_metrics = full_gallery_metrics(right[indices], relevance[indices])
        for name in names:
            deltas[name].append(left_metrics[name] - right_metrics[name])
    expected = {
        name: {
            "lower": float(torch.quantile(torch.tensor(values), 0.025)),
            "median": float(torch.quantile(torch.tensor(values), 0.5)),
            "upper": float(torch.quantile(torch.tensor(values), 0.975)),
        }
        for name, values in deltas.items()
    }
    assert optimized["rankings_recomputed_per_replicate"] is False
    for name in names:
        assert optimized["intervals_95"][name] == pytest.approx(expected[name])
