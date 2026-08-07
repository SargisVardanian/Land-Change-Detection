from __future__ import annotations

import torch

from qcpr_temporal_siglip import (
    TemporalSigLIP,
    TemporalSigLIPConfig,
    TemporalSoftChangeMap,
    build_relevance_masks,
    direct_retrieval_metrics,
    positive_weights_from_grades,
    rank_pairs,
    symmetric_mult_positive_clip_loss,
)
from qcpr_temporal_siglip.trainer import build_optimizer


def _config() -> TemporalSigLIPConfig:
    return TemporalSigLIPConfig(
        hidden_size=16,
        patch_tokens=4,
        patch_grid=2,
        max_frames=4,
        attention_heads=4,
    ).validate()


def _features(pairs: int = 3, queries: int = 3, frames: int = 2):
    torch.manual_seed(19)
    return (
        torch.randn(pairs, frames, 4, 16),
        torch.randn(queries, 5, 16),
        torch.randn(queries, 16),
        torch.ones(queries, 5, dtype=torch.bool),
    )


def test_temporalsiglip_is_direct_and_has_no_evidence_branch():
    model = TemporalSigLIP(config=_config())
    frame_tokens, text_tokens, text_embeddings, text_mask = _features()
    output = model.forward_from_features(
        frame_tokens, text_tokens, text_embeddings, text_mask
    )
    expected = model.effective_logit_scale * (
        output.text_embedding @ output.pair_embedding.transpose(0, 1)
    )
    assert torch.allclose(output.score_matrix, expected)
    assert output.pair_embedding.shape == (3, 16)
    assert output.temporal_tokens.shape == (3, 2, 4, 16)
    assert not hasattr(model, "evidence_bottleneck")
    assert not hasattr(model, "relevance_model")


def test_temporalsiglip_supports_three_frames_and_query_independent_pairs():
    model = TemporalSigLIP(config=_config()).eval()
    frame_tokens, text_tokens, text_embeddings, text_mask = _features(frames=3)
    first = model.encode_pair_from_features(frame_tokens)
    second = model.encode_pair_from_features(frame_tokens)
    assert first.temporal_tokens.shape == (3, 3, 4, 16)
    assert torch.equal(first.pair_embedding, second.pair_embedding)


def test_score_path_promotes_mixed_bfloat16_and_float32_features():
    model = TemporalSigLIP(config=_config())
    text = torch.randn(2, 16, dtype=torch.bfloat16)
    pair = torch.randn(3, 16, dtype=torch.float32)
    scores = model.score_embeddings(text, pair)
    assert scores.shape == (2, 3)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()


def test_symmetric_clip_loss_uses_both_directions_and_multi_positive_masks():
    scores = torch.tensor(
        [[4.0, 1.0, 0.0], [0.5, 3.0, 1.0], [0.0, 0.5, 2.5]],
        requires_grad=True,
    )
    grades = torch.tensor([[3, 2, 0], [0, 3, 0], [0, 0, 3]])
    exact = build_relevance_masks(grades, mode="exact")
    semantic = build_relevance_masks(grades, mode="semantic")
    assert exact.positive.tolist() == [[True, False, False], [False, True, False], [False, False, True]]
    assert exact.ignored.tolist() == [[False, True, False], [False, False, False], [False, False, False]]
    assert semantic.positive.tolist() == [[True, True, False], [False, True, False], [False, False, True]]
    weights = positive_weights_from_grades(grades, semantic, grade1_weight=0.25)
    assert weights.shape == grades.shape
    loss = symmetric_mult_positive_clip_loss(scores, semantic.positive, semantic.ignored)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_pair_to_text_requires_each_pair_to_have_a_positive_caption():
    scores = torch.randn(2, 3)
    positives = torch.tensor([[True, False, False], [False, True, False]])
    try:
        symmetric_mult_positive_clip_loss(scores, positives)
    except ValueError as exc:
        assert "every row" in str(exc)
    else:
        raise AssertionError("pair-to-text must reject a gallery pair without a caption")


def test_localization_is_post_retrieval_and_has_one_trainable_scalar():
    config = _config()
    localizer = TemporalSoftChangeMap(config)
    temporal = torch.randn(2, 2, 4, 16, requires_grad=True)
    text = torch.randn(2, 16)
    output = localizer(temporal, text)
    assert output.map_logits.shape == (2, 1, 2, 2)
    assert output.map_probabilities.shape == (2, 1, 2, 2)
    assert sum(parameter.numel() for parameter in localizer.parameters()) == 1
    output.map_logits.sum().backward()
    assert temporal.grad is not None
    assert torch.isfinite(temporal.grad).all()


def test_localization_promotes_mixed_bfloat16_and_float32_features():
    config = _config()
    localizer = TemporalSoftChangeMap(config)
    temporal = torch.randn(2, 2, 4, 16, dtype=torch.bfloat16)
    text = torch.randn(2, 16, dtype=torch.float32)
    output = localizer(temporal, text)
    assert output.map_logits.dtype == torch.bfloat16
    assert torch.isfinite(output.map_logits).all()


def test_direct_evaluator_has_top10_primary_and_no_reranker_contract():
    scores = torch.tensor([[4.0, 3.0, 2.0], [1.0, 5.0, 0.0]])
    exact = torch.tensor([[True, False, False], [False, True, False]])
    indices = rank_pairs(scores, top_k=2)
    assert indices.tolist() == [[0, 1], [1, 0]]
    metrics = direct_retrieval_metrics(scores, exact)
    assert metrics["hit_at_10"] == 1.0
    assert metrics["mrr_full"] == 1.0


def test_stage_a_optimizer_contains_only_temporal_pair_and_scale():
    model = TemporalSigLIP(config=_config())
    optimizer, scheduler, report = build_optimizer(model, stage="A", total_steps=16)
    assert scheduler is not None
    names = {group["name"] for group in report["groups"]}
    assert names == {"logit_scale", "temporal_pair"}
    assert all(parameter.requires_grad for parameter in model.temporal_pair.parameters())
    assert model.logit_scale.requires_grad
    assert optimizer.param_groups
