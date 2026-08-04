from __future__ import annotations

import torch

from land_change_detection.models.qcpr_hrg import (
    QCPRHRGConfig,
    QCPRHierarchicalRetriever,
)


def _model() -> QCPRHierarchicalRetriever:
    return QCPRHierarchicalRetriever(
        QCPRHRGConfig(
            visual_dim=12,
            retrieval_dim=8,
            token_dim=8,
            token_grid=2,
            context_grid=1,
            slot_count=3,
            slot_heads=2,
            pair_layers=2,
            pair_heads=3,
            token_adapter_layers=2,
            dropout=0.0,
            evidence_topk=2,
        )
    ).eval()


def test_hrg_supports_t2_and_long_series_with_one_global_vector() -> None:
    torch.manual_seed(7)
    model = _model()
    frames_t2 = torch.randn(2, 2, 16, 12)
    frames_t4 = torch.randn(2, 4, 16, 12)
    visual_t2 = model.encode_visual(
        frames_t2,
        grid_shape=(4, 4),
        time_intervals=torch.tensor([[1.0], [2.0]]),
    )
    visual_t4 = model.encode_visual(
        frames_t4,
        grid_shape=(4, 4),
        time_intervals=torch.tensor([[1.0, 2.0, 4.0], [2.0, 2.0, 1.0]]),
    )
    assert visual_t2.global_pair_vector.shape == (2, 8)
    assert visual_t2.change_slot_vectors.shape == (2, 3, 8)
    assert visual_t2.dense_change_tokens.shape == (2, 4, 8)
    assert visual_t2.pyramid_tokens.shape == (2, 5, 8)
    assert visual_t4.global_pair_vector.shape == (2, 8)
    assert visual_t2.metadata["global_indexable"] is True
    assert visual_t2.metadata["full_gallery_cross_attention"] is False
    assert not torch.allclose(
        visual_t2.global_pair_vector,
        visual_t4.global_pair_vector,
    )


def test_hrg_zero_gates_preserve_global_ann_score_and_late_branches_are_candidate_only() -> None:
    torch.manual_seed(11)
    model = _model()
    frames = torch.randn(2, 2, 16, 12)
    visual = model.encode_visual(frames, grid_shape=(4, 4))
    query_global = torch.randn(3, 8)
    query_tokens = torch.randn(3, 5, 8)
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False], [True, True, True, True, False]]
    )
    output = model.candidate_scores(query_global, query_tokens, valid, visual)
    assert output.scores.shape == (3, 2)
    assert output.evidence_logits.shape == (3, 2, 4)
    assert output.evidence_map.shape == (3, 2, 2, 2)
    assert torch.allclose(output.scores, output.global_scores)
    with torch.no_grad():
        model.late_gate.fill_(1.0)
        model.evidence_gate.fill_(1.0)
    reranked = model.rerank_candidates(
        query_global,
        query_tokens,
        valid,
        visual,
        torch.tensor([[0, 1], [1, 0], [0, 1]]),
    )
    assert reranked.scores.shape == (3, 2)
    assert reranked.visual.metadata["candidate_count"] == 2
    assert not torch.allclose(reranked.scores, reranked.global_scores)


def test_hrg_evidence_is_query_sensitive_and_temporal_intervals_are_used() -> None:
    torch.manual_seed(13)
    model = _model()
    frames = torch.randn(1, 2, 16, 12)
    short = model.encode_visual(frames, grid_shape=(4, 4), time_intervals=torch.tensor([[1.0]]))
    long = model.encode_visual(frames, grid_shape=(4, 4), time_intervals=torch.tensor([[9.0]]))
    assert not torch.allclose(short.pyramid_tokens, long.pyramid_tokens)
    queries = torch.randn(2, 4, 8)
    globals_ = torch.randn(2, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    evidence = model.candidate_scores(globals_, queries, valid, short)
    swapped = model.candidate_scores(globals_.flip(0), queries.flip(0), valid, short)
    assert not torch.allclose(evidence.evidence_logits, swapped.evidence_logits)
