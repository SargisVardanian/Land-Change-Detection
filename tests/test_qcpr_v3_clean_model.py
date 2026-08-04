from __future__ import annotations

from dataclasses import replace

import torch

from qcpr_v3.config.schema import EvidenceConfig, QCPRConfig, TemporalConfig, TextConfig, TrainingConfig
from qcpr_v3.data.batch import assert_equal_schedule, schedule_hash
from qcpr_v3.data.contracts import TemporalMetadata, assert_mask_free_record
from qcpr_v3.evaluation.long_series import pad_temporal_tokens, reverse_temporal_tokens
from qcpr_v3.models import QCPRV3Model
from qcpr_v3.models.evidence_bottleneck import sparsemax
from qcpr_v3.training.objective import UnifiedListwiseLoss


def _config() -> QCPRConfig:
    return QCPRConfig(
        temporal=TemporalConfig(
            native_dim=16,
            hidden_dim=512,
            heads=8,
            blocks=1,
            change_slots=2,
            dropout=0.0,
            max_frames=8,
        ),
        text=TextConfig(input_dim=16, hidden_dim=512, adapter_layers=1, heads=8, dropout=0.0),
        evidence=EvidenceConfig(sparse_normalizer="softmax", pairwise_query_chunk=2),
        training=TrainingConfig(max_steps=1),
    )


def _metadata(batch: int, time_count: int) -> TemporalMetadata:
    timestamps = torch.arange(time_count, dtype=torch.float32).view(1, -1).expand(batch, -1)
    return TemporalMetadata(
        timestamps=timestamps,
        delta_times=timestamps - timestamps[:, :1],
        frame_ids=torch.arange(time_count).view(1, -1).expand(batch, -1),
    )


def _inputs(pair_count: int = 3, query_count: int = 2, time_count: int = 2):
    torch.manual_seed(13)
    config = _config()
    native = torch.randn(pair_count, time_count, 4, 16)
    coordinates = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).view(1, 4, 2).expand(pair_count, -1, -1)
    query_native = torch.randn(query_count, 5, 16)
    return config, native, coordinates, _metadata(pair_count, time_count), query_native


def test_model_supports_t2_and_t3_and_is_query_independent():
    config, native, coordinates, metadata, query_native = _inputs()
    model = QCPRV3Model(config).eval()
    visual = model.encode_visual(native, coordinates, metadata)
    query = model.encode_query(query_native, torch.ones(2, 5, dtype=torch.bool))
    scored = model.score(query, visual)
    assert visual.sequence_cls.shape == (3, 512)
    assert visual.frame_cls.shape == (3, 2, 512)
    assert scored.scores.shape == (2, 3)
    assert scored.evidence_weights is not None
    assert scored.evidence_weights.shape == (2, 3, 2, 4)
    three = torch.randn(3, 3, 4, 16)
    visual_three = model.encode_visual(three, coordinates, _metadata(3, 3))
    assert visual_three.frame_cls.shape == (3, 3, 512)
    assert torch.isfinite(scored.scores).all()


def test_zero_gates_reduce_score_to_global_ann():
    config, native, coordinates, metadata, query_native = _inputs()
    model = QCPRV3Model(config).eval()
    visual = model.encode_visual(native, coordinates, metadata)
    query = model.encode_query(query_native, torch.ones(2, 5, dtype=torch.bool))
    with torch.no_grad():
        model.relevance_model.evidence_gate.zero_()
        model.relevance_model.slot_gate.zero_()
        model.relevance_model.late_gate.zero_()
    scored = model.score(query, visual)
    ann = model.ann_scores(query, visual)
    assert torch.allclose(scored.scores, ann, atol=1e-6, rtol=1e-6)


def test_evidence_path_has_gradient_and_one_primary_loss():
    config, native, coordinates, metadata, query_native = _inputs()
    model = QCPRV3Model(config)
    visual = model.encode_visual(native, coordinates, metadata)
    query = model.encode_query(query_native, torch.ones(2, 5, dtype=torch.bool))
    grades = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long)
    output = model.score(query, visual)
    loss = UnifiedListwiseLoss(temperature=0.07)(output.scores, grades)
    assert loss.primary_scalar_count == 1
    loss.loss.backward()
    gradient = model.evidence_bottleneck.query_projection.weight.grad
    assert gradient is not None
    assert float(gradient.norm()) > 0.0


def test_temporal_double_reversal_restores_order():
    config, native, coordinates, metadata, _ = _inputs(time_count=3)
    model = QCPRV3Model(config).eval()
    original = model.encode_visual(native, coordinates, metadata)
    reversed_once = model.encode_visual(native.flip(1), coordinates, _metadata(3, 3))
    restored = model.encode_visual(native.flip(1).flip(1), coordinates, metadata)
    assert not torch.allclose(original.sequence_cls, reversed_once.sequence_cls)
    assert torch.allclose(original.sequence_cls, restored.sequence_cls, atol=1e-6, rtol=1e-6)


def test_ann_is_deterministic_and_rerank_touches_only_top_k():
    config, native, coordinates, metadata, query_native = _inputs(pair_count=4, query_count=1)
    model = QCPRV3Model(config).eval()
    visual = model.encode_visual(native, coordinates, metadata)
    query = model.encode_query(query_native[:1], torch.ones(1, 5, dtype=torch.bool))
    first = model.ann_scores(query, visual)
    second = model.ann_scores(query, visual)
    assert torch.equal(first, second)
    indices, reranked = model.rerank(query, visual, top_k=2)
    assert indices.shape == (1, 2)
    assert reranked.scores.shape == (1, 2)
    assert reranked.evidence_weights is not None
    assert reranked.evidence_weights.shape == (1, 2, 2, 4)
    assert bool((indices < 4).all())


def test_evidence_map_is_query_conditioned_and_causal_in_score():
    config, native, coordinates, metadata, query_native = _inputs(pair_count=3, query_count=2)
    model = QCPRV3Model(config).eval()
    visual = model.encode_visual(native, coordinates, metadata)
    query = model.encode_query(query_native, torch.ones(2, 5, dtype=torch.bool))
    with torch.no_grad():
        model.relevance_model.evidence_gate.fill_(1.0)
    scored = model.score(query, visual)
    assert scored.evidence_weights is not None
    assert not torch.allclose(scored.evidence_weights[0], scored.evidence_weights[1])
    zero_dense = replace(visual, dense_tokens=torch.zeros_like(visual.dense_tokens))
    zeroed = model.score(query, zero_dense)
    assert not torch.allclose(scored.scores, zeroed.scores)


def test_variable_length_mask_and_small_displacement_contract():
    config, native, coordinates, metadata, _ = _inputs(pair_count=2, time_count=3)
    model = QCPRV3Model(config).eval()
    frame_mask = torch.tensor([[True, True, False], [True, True, True]])
    token_mask = frame_mask[:, :, None].expand(2, 3, 4).clone()
    token_mask[0, 1, 0] = False
    shifted = coordinates + torch.tensor([0.25, -0.25])
    output = model.encode_visual(native, shifted, metadata, frame_mask=frame_mask, token_mask=token_mask)
    assert output.dense_tokens.shape == (2, 3, 4, 512)
    assert torch.isfinite(output.dense_tokens).all()


def test_sparsemax_is_a_simplex_and_schedule_is_deterministic():
    values = torch.tensor([[2.0, 0.0, -1.0]])
    weights = sparsemax(values)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    first = schedule_hash(["a", "b"], ["q1", "q2"])
    assert_equal_schedule(first, dict(first))


def test_long_series_padding_and_mask_free_contract():
    padded, mask = pad_temporal_tokens([torch.ones(2, 3), torch.zeros(4, 3)])
    assert padded.shape == (2, 4, 3)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    sequence = torch.arange(12).reshape(1, 3, 4)
    assert torch.equal(reverse_temporal_tokens(sequence), sequence.flip(1))
    assert_mask_free_record({"query_id": "q", "text": "a building appeared"})
    try:
        assert_mask_free_record({"query_id": "q", "mask_path": "labels/a.png"})
    except ValueError:
        pass
    else:
        raise AssertionError("mask path was accepted into training")
