import json

import pytest
import torch
from run_qcpr_siglip2_real_smoke import build_relevance

from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.loader import make_exact_batches
from qcpr_siglip2.evaluation.evidence import effective_token_count, evidence_entropy
from qcpr_siglip2.evaluation.reranking import (
    rerank_candidate_indices,
    select_topk_candidates,
)
from qcpr_siglip2.evaluation.retrieval import (
    candidate_hit_at_k,
    mean_rank,
    median_rank,
    mrr_at_k,
    mrr_full,
    multi_positive_recall_at_k,
)
from qcpr_siglip2.models.evidence import EvidenceBottleneck
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.models.temporal import TemporalTransformerAdapter
from qcpr_siglip2.training.objective import multi_positive_listwise_loss
from qcpr_siglip2.training.optimizer import build_adamw


def _features(q=4, p=3, t=2, n=256, l=7, d=768):
    torch.manual_seed(7)
    return (
        torch.randn(p, t, n, d),
        torch.randn(p, t, d),
        torch.randn(q, l, d),
        torch.randn(q, d),
        torch.ones(q, l, dtype=torch.bool),
    )


def test_config_is_minimal_two_layer_contract():
    cfg = Siglip2TemporalConfig()
    assert cfg.validate().temporal_layers == 2
    assert cfg.mlp_size == 4 * cfg.hidden_size


def test_model_outputs_causal_evidence_and_map():
    model = Siglip2TemporalRetrievalModel(None)
    out = model.forward_from_features(*_features(q=3, p=2))
    assert out.score_matrix.shape == (3, 2)
    assert out.evidence.evidence_map.shape == (3, 2, 2, 16, 16)
    assert out.evidence.evidence_vector.requires_grad
    out.score_matrix.sum().backward()
    assert model.evidence_bottleneck.raw_gate.grad is not None
    assert model.temporal_adapter.blocks[0].attn_scale.grad is not None


def test_query_swap_changes_evidence_and_score():
    model = Siglip2TemporalRetrievalModel(None)
    out = model.forward_from_features(*_features(q=2, p=1))
    assert not torch.allclose(
        out.evidence.evidence_map[0], out.evidence.evidence_map[1]
    )
    assert not torch.allclose(out.score_matrix[0], out.score_matrix[1])


def test_evidence_zeroing_changes_final_score():
    model = Siglip2TemporalRetrievalModel(None)
    frame_tokens, frame_embeddings, text_tokens, text_embeddings, text_mask = _features(
        q=2, p=2
    )
    normal = model.forward_from_features(
        frame_tokens, frame_embeddings, text_tokens, text_embeddings, text_mask
    )
    pair = torch.nn.functional.normalize(normal.pair_cls.unsqueeze(0), dim=-1)
    zero_score = (
        torch.einsum("qd,qpd->qp", normal.text_embedding, pair)
        / model.retrieval_temperature
    )
    assert not torch.allclose(normal.score_matrix, zero_score)


def test_one_primary_scalar_loss_and_metrics():
    scores = torch.tensor([[4.0, 3.0, 1.0], [2.0, 1.0, 0.0]])
    positives = torch.tensor([[True, True, False], [False, True, False]])
    loss = multi_positive_listwise_loss(scores, positives)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert candidate_hit_at_k(scores, positives, 1) == 0.5
    assert multi_positive_recall_at_k(scores, positives, 1) == 0.25
    assert mrr_full(scores, positives) == 0.75
    assert mrr_at_k(scores, positives, 1) == 0.5


def test_hit_and_recall_differ_for_multiple_positives():
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    positives = torch.tensor([[True, True, False]])
    assert candidate_hit_at_k(scores, positives, 1) == 1.0
    assert multi_positive_recall_at_k(scores, positives, 1) == 0.5


def test_rank_diagnostics_are_explicit():
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    relevance = torch.tensor([[False, True, False]])
    assert mean_rank(scores, relevance) == 2.0
    assert median_rank(scores, relevance) == 2.0


def test_evidence_bottleneck_rejects_no_valid_text():
    with pytest.raises(ValueError, match="at least one"):
        EvidenceBottleneck(Siglip2TemporalConfig())(
            torch.randn(1, 2, 768),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.randn(1, 512, 768),
            frame_count=2,
            patch_count=256,
        )


def test_blockwise_evidence_matches_unchunked_values_and_gradients():
    torch.manual_seed(17)
    full_cfg = Siglip2TemporalConfig(
        evidence_query_chunk_size=64, evidence_pair_chunk_size=64
    )
    block_cfg = Siglip2TemporalConfig(
        evidence_query_chunk_size=1, evidence_pair_chunk_size=1
    )
    full = EvidenceBottleneck(full_cfg)
    block = EvidenceBottleneck(block_cfg)
    block.load_state_dict(full.state_dict())
    text = torch.randn(3, 5, 768, requires_grad=True)
    visual = torch.randn(2, 512, 768, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    full_out = full(text, mask, visual, frame_count=2, patch_count=256)
    full_loss = (
        full_out.evidence_logits.square().mean()
        + full_out.evidence_vector.square().mean()
    )
    full_loss.backward()
    text_grad = text.grad.detach().clone()
    visual_grad = visual.grad.detach().clone()
    text.grad = None
    visual.grad = None
    block_out = block(text, mask, visual, frame_count=2, patch_count=256)
    block_loss = (
        block_out.evidence_logits.square().mean()
        + block_out.evidence_vector.square().mean()
    )
    block_loss.backward()
    assert torch.allclose(
        full_out.evidence_logits, block_out.evidence_logits, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        full_out.evidence_weights, block_out.evidence_weights, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        full_out.evidence_vector, block_out.evidence_vector, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(text_grad, text.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(visual_grad, visual.grad, atol=2e-5, rtol=2e-5)


def test_global_stage_one_score_is_unscaled_cosine():
    output = Siglip2TemporalRetrievalModel(None).forward_from_features(
        *_features(q=2, p=3)
    )
    expected = output.text_embedding @ output.pair_cls.transpose(0, 1)
    assert torch.allclose(output.global_score_matrix, expected)


def test_deterministic_sampler_rotates_captions_without_pair_weight_drift():
    rows = [
        {"canonical_pair_id": "p0", "caption_id": "p0q0"},
        {"canonical_pair_id": "p0", "caption_id": "p0q1"},
        {"canonical_pair_id": "p0", "caption_id": "p0q2"},
        {"canonical_pair_id": "p1", "caption_id": "p1q0"},
        {"canonical_pair_id": "p1", "caption_id": "p1q1"},
        {"canonical_pair_id": "p1", "caption_id": "p1q2"},
    ]
    first = make_exact_batches(rows, physical_batch_size=2, epoch=0, seed=3)[0]
    second = make_exact_batches(rows, physical_batch_size=2, epoch=0, seed=3)[0]
    rotated = make_exact_batches(rows, physical_batch_size=2, epoch=1, seed=3)[0]
    assert first.query_ids == second.query_ids
    assert first.pair_ids == second.pair_ids
    assert len(first.query_ids) == 4
    assert set(first.pair_ids) == {"p0", "p1"}
    assert first.query_ids != rotated.query_ids


def test_reranking_rejects_row_duplicate_candidates():
    with pytest.raises(ValueError, match="duplicate"):
        rerank_candidate_indices(
            torch.tensor([[1, 1, 2]]), torch.tensor([[0.1, 0.2, 0.3]])
        )


def test_reranking_selects_and_orders_candidates():
    global_scores = torch.tensor([[0.2, 0.9, 0.4, 0.1]])
    candidates = select_topk_candidates(global_scores, 3)
    assert candidates.tolist() == [[1, 2, 0]]
    assert rerank_candidate_indices(
        candidates, torch.tensor([[0.1, 0.8, 0.2]])
    ).tolist() == [[2, 0, 1]]


def test_manifest_loader_keeps_generic_no_change_out_of_exact_pair_view(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "dataset_name": "levir_mci",
            "canonical_pair_id": "p0",
            "caption_id": "q0",
            "t1_path": "a",
            "t2_path": "b",
            "caption": "no change",
            "query_scope": "generic_no_change",
            "split": "train",
        },
        {
            "dataset_name": "levir_mci",
            "canonical_pair_id": "p0",
            "caption_id": "q1",
            "t1_path": "a",
            "t2_path": "b",
            "caption": "a building appeared",
            "query_scope": "exact_pair",
            "split": "train",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    from qcpr_siglip2.data.manifest import load_exact_core_rows, load_exact_pair_rows

    assert len(load_exact_core_rows(path)) == 2
    assert [row["caption_id"] for row in load_exact_pair_rows(path)] == ["q1"]


def test_phase_a_optimizer_scope_is_temporal_only():
    model = Siglip2TemporalRetrievalModel(None)
    optimizer, report, scheduler = build_adamw(model, phase="A", total_steps=256)
    assert report["phase"] == "A"
    assert report["trainable_parameter_count"] > 0
    assert all(
        not name.startswith("backbone")
        for group in report["groups"]
        for name in group["parameter_names"]
    )
    assert len(optimizer.param_groups) == len(report["groups"])
    optimizer.step()
    scheduler.step()


def test_smoke_relevance_does_not_infer_positives_from_text_collision():
    rows = [
        {
            "caption_id": "q0",
            "canonical_pair_id": "p0",
            "caption": "the same text",
            "positive_pair_ids": ["p0"],
            "ignored_pair_ids": [],
        },
        {
            "caption_id": "q1",
            "canonical_pair_id": "p1",
            "caption": "the same text",
            "positive_pair_ids": ["p1"],
            "ignored_pair_ids": [],
        },
    ]
    pairs = [{"canonical_pair_id": "p0"}, {"canonical_pair_id": "p1"}]
    positive, ignored, meta = build_relevance(rows, pairs, 1, torch.device("cpu"))
    assert positive.tolist() == [[True, False], [False, True]]
    assert not ignored.any()
    assert meta["multi_positive_queries"] == 0
    assert meta["text_collision_not_used_as_positive"] is True


def test_temporal_metadata_projection_accepts_bfloat16_tokens():
    frame_tokens, frame_embeddings, *_ = _features(q=2, p=2)
    adapter = TemporalTransformerAdapter(Siglip2TemporalConfig()).bfloat16()
    output = adapter(
        frame_tokens.bfloat16(),
        frame_embeddings.bfloat16(),
        timestamps=torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.bfloat16),
    )
    assert output.pair_cls.dtype == torch.bfloat16


def test_temporal_adapter_supports_variable_sequence_lengths_and_reversal():
    torch.manual_seed(23)
    adapter = TemporalTransformerAdapter(Siglip2TemporalConfig(max_frames=4))
    frame_tokens = torch.randn(1, 3, 256, 768)
    frame_embeddings = torch.randn(1, 3, 768)
    forward = adapter(frame_tokens, frame_embeddings)
    reversed_output = adapter(frame_tokens.flip(1), frame_embeddings.flip(1))
    restored = adapter(frame_tokens.flip(1).flip(1), frame_embeddings.flip(1).flip(1))
    assert forward.pair_cls.shape == (1, 768)
    assert forward.temporal_patch_tokens.shape == (1, 768, 768)
    assert not torch.allclose(forward.pair_cls, reversed_output.pair_cls)
    assert torch.allclose(forward.pair_cls, restored.pair_cls)


def test_evidence_diagnostics_report_entropy_and_effective_tokens():
    weights = torch.full((2, 4), 0.25)
    assert torch.allclose(evidence_entropy(weights), torch.full((2,), 2.0))
    assert torch.allclose(effective_token_count(weights), torch.full((2,), 4.0))
