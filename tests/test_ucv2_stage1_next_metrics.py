from __future__ import annotations

import torch
from torch.nn import functional as F

from land_change_detection.models.retrieval_heads import stable_caption_group_ids
from ucv2_retrieval_metrics import RetrievalCorpus, compute_retrieval_metrics, compute_retrieval_ranks


def _tie_corpus() -> RetrievalCorpus:
    return RetrievalCorpus(
        pair_embeddings=F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ]
            ),
            dim=-1,
        ),
        text_embeddings=F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, 0.5],
                ]
            ),
            dim=-1,
        ),
        caption_to_pair=torch.tensor([0, 1, 2, 3]),
        caption_group_ids=stable_caption_group_ids(["same", "same", "other", "third"]),
        pair_ids=["pair-b", "pair-a", "pair-c", "pair-d"],
        captions=["same", "same", "other", "third"],
        pair_mask_fractions=torch.tensor([0.0, 0.0, 0.1, 0.02]),
        encode_seconds=1.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
    )


def _stable_metric_subset(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key.endswith("R@1")
        or key.endswith("R@5")
        or key.endswith("R@10")
        or key in {"MRR", "mean_rank", "exact_pair_MRR", "exact_pair_mean_rank", "num_queries", "num_candidates", "positive_count_min", "positive_count_max"}
    }


def test_extended_metrics_report_exact_frequency_and_mask_strata():
    pair_embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]),
        dim=-1,
    )
    text_embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dim=-1,
    )
    corpus = RetrievalCorpus(
        pair_embeddings=pair_embeddings,
        text_embeddings=text_embeddings,
        caption_to_pair=torch.tensor([0, 2, 1]),
        caption_group_ids=stable_caption_group_ids(["same", "same", "A new building appeared"]),
        pair_ids=["a", "b", "c"],
        captions=["same", "same", "A new building appeared"],
        pair_mask_fractions=torch.tensor([0.0, 0.1, 0.005]),
        encode_seconds=1.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
    )
    metrics, similarities = compute_retrieval_metrics(corpus)
    assert similarities.shape == (3, 3)
    assert metrics["text_to_pair_R@1"] == 1.0
    assert metrics["exact_pair_R@1"] < 1.0
    assert metrics["rare_caption_count"] == 2
    assert metrics["unique_caption_count"] == 1
    assert metrics["mask_no_change_count"] == 1
    assert metrics["mask_small_change_count"] == 1
    assert metrics["mask_large_change_count"] == 1
    assert metrics["appeared_count"] == 1
    assert metrics["changed_count"] == 1
    assert metrics["disappeared_empty"] is True
    assert metrics["text_to_pair_R@1"] <= metrics["text_to_pair_R@5"] <= metrics["text_to_pair_R@10"]
    assert metrics["num_queries"] == 3
    assert metrics["num_candidates"] == 3
    assert metrics["positive_count_min"] >= 1
    assert "rank_fingerprint" in metrics


def test_repeated_retrieval_metrics_are_deterministic_and_do_not_mutate_inputs():
    corpus = _tie_corpus()
    before_pairs = corpus.pair_embeddings.clone()
    before_texts = corpus.text_embeddings.clone()
    first_ranks = compute_retrieval_ranks(corpus)
    second_ranks = compute_retrieval_ranks(corpus)
    first_metrics, _ = compute_retrieval_metrics(corpus)
    second_metrics, _ = compute_retrieval_metrics(corpus)
    assert torch.equal(first_ranks.ranked_candidate_indices, second_ranks.ranked_candidate_indices)
    assert torch.equal(first_ranks.duplicate_aware_ranks, second_ranks.duplicate_aware_ranks)
    assert torch.equal(first_ranks.exact_pair_ranks, second_ranks.exact_pair_ranks)
    assert _stable_metric_subset(first_metrics) == _stable_metric_subset(second_metrics)
    assert torch.equal(corpus.pair_embeddings, before_pairs)
    assert torch.equal(corpus.text_embeddings, before_texts)


def test_chunked_retrieval_matches_full_matrix_for_multiple_chunk_sizes():
    corpus = _tie_corpus()
    full = compute_retrieval_ranks(corpus)
    for query_chunk, candidate_chunk in [(1, 1), (2, 3), (3, 2), (99, 99)]:
        chunked = compute_retrieval_ranks(
            corpus,
            query_chunk_size=query_chunk,
            candidate_chunk_size=candidate_chunk,
        )
        assert torch.equal(chunked.similarities, full.similarities)
        assert torch.equal(chunked.ranked_candidate_indices, full.ranked_candidate_indices)
        assert torch.equal(chunked.duplicate_aware_ranks, full.duplicate_aware_ranks)
        assert torch.equal(chunked.exact_pair_ranks, full.exact_pair_ranks)


def test_metrics_are_derived_from_same_rank_tensor_and_are_monotonic():
    corpus = _tie_corpus()
    ranks = compute_retrieval_ranks(corpus)
    metrics, _ = compute_retrieval_metrics(corpus)
    assert metrics["text_to_pair_R@1"] == float((ranks.duplicate_aware_ranks <= 1).float().mean().item())
    assert metrics["text_to_pair_R@5"] == float((ranks.duplicate_aware_ranks <= 5).float().mean().item())
    assert metrics["text_to_pair_R@10"] == float((ranks.duplicate_aware_ranks <= 10).float().mean().item())
    assert metrics["text_to_pair_R@1"] <= metrics["text_to_pair_R@5"] <= metrics["text_to_pair_R@10"]


def test_candidate_shuffle_with_stable_ids_preserves_ranks_under_ties():
    corpus = _tie_corpus()
    base_ranks = compute_retrieval_ranks(corpus)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    shuffled = RetrievalCorpus(
        pair_embeddings=corpus.pair_embeddings[permutation],
        text_embeddings=corpus.text_embeddings,
        caption_to_pair=inverse[corpus.caption_to_pair],
        caption_group_ids=corpus.caption_group_ids,
        pair_ids=[corpus.pair_ids[index] for index in permutation.tolist()],
        captions=corpus.captions,
        pair_mask_fractions=corpus.pair_mask_fractions[permutation],
        encode_seconds=1.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
    )
    shuffled_ranks = compute_retrieval_ranks(shuffled)
    assert torch.equal(shuffled_ranks.duplicate_aware_ranks, base_ranks.duplicate_aware_ranks)
    assert torch.equal(shuffled_ranks.exact_pair_ranks, base_ranks.exact_pair_ranks)


def test_tied_similarity_scores_use_stable_pair_id_tiebreaking():
    corpus = _tie_corpus()
    ranks = compute_retrieval_ranks(corpus)
    tied_candidates = torch.tensor([0, 1])
    tied_order = ranks.ranked_candidate_indices[0, :2]
    expected = tied_candidates[torch.argsort(ranks.candidate_tie_keys[tied_candidates], stable=True)]
    assert torch.equal(tied_order, expected)


def test_tie_and_margin_diagnostics_are_reported():
    corpus = _tie_corpus()
    metrics, _ = compute_retrieval_metrics(corpus)
    assert metrics["exact_tie_count"] >= 2
    assert metrics["near_tie_count_eps_1e-6"] >= 2
    assert metrics["top1_top2_margin_count"] == 4
    assert "top1_top2_margin_mean" in metrics
    assert "best_positive_minus_best_negative_mean" in metrics
    assert metrics["tie_aware_optimistic_R@1"] >= metrics["tie_aware_pessimistic_R@1"]


def test_misaligned_retrieval_corpus_is_rejected():
    corpus = _tie_corpus()
    bad = RetrievalCorpus(
        pair_embeddings=corpus.pair_embeddings,
        text_embeddings=corpus.text_embeddings,
        caption_to_pair=torch.tensor([0, 1, 2, 99]),
        caption_group_ids=corpus.caption_group_ids,
        pair_ids=corpus.pair_ids,
        captions=corpus.captions,
        pair_mask_fractions=corpus.pair_mask_fractions,
        encode_seconds=1.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
    )
    try:
        compute_retrieval_ranks(bad)
    except ValueError as exc:
        assert "out-of-range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range caption_to_pair to be rejected")
