from __future__ import annotations

import math

import torch

from land_change_detection.models.qcpr_v3 import (
    QCPRV3Config,
    QCPRV3GenericGrounding,
    apply_two_stage_reranking,
    candidate_recall_at_n,
    stable_global_top_n,
)
from land_change_detection.models.qcpr_v3_adapters import (
    CanonicalV3Inputs,
    evaluator_score,
    faithful_mask_score,
    renderer_score,
    trainer_score,
)


def _inputs() -> tuple[QCPRV3GenericGrounding, CanonicalV3Inputs]:
    torch.manual_seed(4)
    config = QCPRV3Config(
        input_dim=16, text_dim=16, hidden_dim=16, heads=4,
        decoder_layers=1, scales=(4, 2), output_size=(32, 32),
        token_top_k=2,
    )
    model = QCPRV3GenericGrounding(config).eval()
    inputs = CanonicalV3Inputs(
        global_query_embeddings=torch.randn(3, 16),
        text_token_embeddings=torch.randn(3, 6, 16),
        text_attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        pair_embeddings=torch.randn(4, 16),
        per_time_tokens=torch.randn(4, 2, 16, 16),
    )
    return model, inputs


def test_generic_v3_shapes_and_no_semantic_specific_heads() -> None:
    model, inputs = _inputs()
    output = model(**inputs.as_kwargs())
    assert output.global_score.shape == (3, 4)
    assert output.local_score.shape == (3, 4)
    assert output.token_patch_score.shape == (3, 4)
    assert output.patch_mask_logits.shape == (3, 4, 16)
    assert output.decoded_mask_logits.shape == (3, 4, 32, 32)
    assert output.mask_mass.shape == (3, 4)
    assert output.mask_validity.shape == (3, 4)
    assert output.temporal_descriptors.shape == (4, 16, 16)
    assert output.slot_activations is None
    names = set(dict(model.named_modules()))
    assert not any(term in name for name in names for term in ("object_head", "direction_head", "location_head", "count_head", "relation_head"))


def test_sparse_mask_prior_initializes_final_logit_bias() -> None:
    model, _ = _inputs()
    expected = math.log(
        model.config.mask_prior_probability / (1.0 - model.config.mask_prior_probability)
    )
    torch.testing.assert_close(
        model.mask_decoder.mask_head[-1].bias.detach(),
        torch.tensor([expected], dtype=model.mask_decoder.mask_head[-1].bias.dtype),
    )


def test_temporal_field_projects_raw_visual_tokens_before_signed_difference() -> None:
    config = QCPRV3Config(
        input_dim=8, visual_source_dim=12, text_dim=8, hidden_dim=8,
        heads=2, decoder_layers=1, scales=(4,), output_size=(8, 8), mask_decoder_dim=8,
    )
    model = QCPRV3GenericGrounding(config)
    raw = torch.randn(2, 2, 16, 12)
    field = model.temporal_field(raw)
    assert field.descriptors.shape == (2, 16, 8)
    assert model.temporal_field.source_projection.weight.grad is None
    field.descriptors.mean().backward()
    assert model.temporal_field.source_projection.weight.grad is not None


def test_grounding_tokens_and_global_query_may_use_different_frozen_encoders() -> None:
    config = QCPRV3Config(
        input_dim=8,
        visual_source_dim=12,
        text_dim=12,
        global_text_dim=8,
        hidden_dim=8,
        heads=2,
        decoder_layers=1,
        scales=(4, 2),
        output_size=(8, 8),
        mask_decoder_dim=8,
    )
    model = QCPRV3GenericGrounding(config)
    output = model.score_query_pair_chunks(
        torch.randn(2, 8),
        torch.randn(2, 5, 12),
        torch.ones(2, 5, dtype=torch.bool),
        torch.randn(3, 8),
        torch.randn(3, 2, 16, 12),
    )
    assert output.global_score.shape == (2, 3)
    assert output.decoded_mask_logits.shape == (2, 3, 8, 8)


def test_temporal_field_preserves_siglip_native_grid() -> None:
    config = QCPRV3Config(
        input_dim=8,
        visual_source_dim=12,
        text_dim=8,
        hidden_dim=8,
        heads=2,
        decoder_layers=1,
        scales=(8, 4),
        output_size=(16, 16),
        mask_decoder_dim=8,
    )
    field = QCPRV3GenericGrounding(config).temporal_field(
        torch.randn(1, 2, 16, 12)
    )
    assert field.scale_slices == ((0, 16, 4),)


def test_all_consumers_share_exact_scores_and_mask_logits() -> None:
    model, inputs = _inputs()
    outputs = [trainer_score(model, inputs), evaluator_score(model, inputs), renderer_score(model, inputs)]
    for output in outputs[1:]:
        assert torch.equal(outputs[0].reranked_score, output.reranked_score)
        assert torch.equal(outputs[0].decoded_mask_logits, output.decoded_mask_logits)
        assert torch.equal(outputs[0].patch_mask_logits, output.patch_mask_logits)
    assert torch.equal(faithful_mask_score(model, inputs), outputs[0].local_score)


def test_b_direct_late_interaction_skips_cross_attention_and_mask_decoder() -> None:
    model, inputs = _inputs()
    output = model.score_query_pair_chunks(**inputs.as_kwargs(), decode_mask=False)
    assert output.decoded_mask_logits.shape == (3, 4, 0, 0)
    assert output.patch_mask_logits.shape == (3, 4, 0)
    torch.testing.assert_close(
        output.reranked_score,
        output.global_score + 0.1 * output.token_patch_score,
    )


def test_multiscale_decoder_keeps_small_local_peak() -> None:
    model, _ = _inputs()
    logits = torch.full((1, 1, 16), -12.0)
    logits[..., 0] = 12.0
    grounded = torch.randn(1, 1, 16, model.config.hidden_dim)
    decoded = model.mask_decoder(logits, grounded, ((0, 16, 4),))
    assert decoded.shape == (1, 1, *model.config.output_size)
    changed = logits.clone(); changed[..., 0] = -12.0
    assert not torch.equal(decoded, model.mask_decoder(changed, grounded, ((0, 16, 4),)))


def test_feature_decoder_consumes_grounded_skip_features() -> None:
    model, _ = _inputs()
    logits = torch.full((1, 1, 16), 0.25)
    grounded = torch.zeros(1, 1, 16, model.config.hidden_dim)
    baseline = model.mask_decoder(logits, grounded, ((0, 16, 4),))
    grounded[..., 0, 0] = 1.0
    changed = model.mask_decoder(logits, grounded, ((0, 16, 4),))
    assert not torch.equal(baseline, changed)


def test_query_modulation_makes_dense_features_text_conditional() -> None:
    model, _ = _inputs()
    patches = torch.randn(1, 16, model.config.hidden_dim)
    tokens = torch.randn(2, 3, model.config.text_dim)
    attention = torch.ones(2, 3, dtype=torch.bool)
    grounded, _, _ = model.grounding_decoder(patches, tokens, attention)
    assert not torch.allclose(grounded[0], grounded[1])


def test_global_query_embedding_directly_conditions_dense_features() -> None:
    model, _ = _inputs()
    patches = torch.randn(1, 16, model.config.hidden_dim)
    shared_tokens = torch.randn(1, 3, model.config.text_dim).expand(2, -1, -1).clone()
    attention = torch.ones(2, 3, dtype=torch.bool)
    direction = torch.linspace(-1.0, 1.0, model.config.hidden_dim)
    global_queries = torch.stack((direction, -direction))
    grounded, _, _ = model.grounding_decoder(patches, shared_tokens, attention, global_queries)
    assert not torch.allclose(grounded[0], grounded[1])


def test_multiscale_fusion_sends_gradient_to_every_refiner() -> None:
    model, _ = _inputs()
    logits = torch.full((1, 1, 16), -12.0, requires_grad=True)
    logits.data[..., 0] = 12.0
    grounded = torch.randn(1, 1, 16, model.config.hidden_dim, requires_grad=True)
    model.mask_decoder(logits, grounded, ((0, 16, 4),)).mean().backward()
    for block in (*model.mask_decoder.lateral_projections, *model.mask_decoder.fusion_blocks):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in block.parameters())


def test_empty_content_tokens_are_finite_and_near_empty_mask_gates_local_score() -> None:
    model, inputs = _inputs()
    empty_content = torch.zeros_like(inputs.text_attention_mask)
    output = model(**(inputs.__class__(**{**inputs.__dict__, "text_content_mask": empty_content}).as_kwargs()))
    assert torch.isfinite(output.token_patch_score).all()
    with torch.no_grad():
        for parameter in model.mask_decoder.parameters():
            parameter.zero_()
        model.mask_decoder.mask_head[-1].bias.fill_(-20.0)
    gated = model(**inputs.as_kwargs())
    assert torch.allclose(gated.local_score, torch.zeros_like(gated.local_score), atol=1e-6)


def test_full_grounder_constant_visual_input_has_no_systematic_border() -> None:
    model, inputs = _inputs()
    constant = torch.zeros_like(inputs.per_time_tokens)
    output = model(**(inputs.__class__(**{**inputs.__dict__, "per_time_tokens": constant}).as_kwargs()))
    probability = output.decoded_mask_logits.sigmoid()
    edge = torch.cat((
        probability[..., 0, :].flatten(), probability[..., -1, :].flatten(),
        probability[..., :, 0].flatten(), probability[..., :, -1].flatten(),
    )).mean()
    center = probability[..., 8:24, 8:24].mean()
    assert float((edge - center).abs().detach()) < 0.10


def test_patch_pooling_logits_are_sampled_from_displayed_mask_field() -> None:
    model, inputs = _inputs()
    output = model(**inputs.as_kwargs())
    expected = model.mask_decoder.sample_patch_logits(
        output.decoded_mask_logits,
        ((0, 16, 4),),
    )
    torch.testing.assert_close(output.patch_mask_logits, expected)


def test_two_stage_selection_is_deterministic_and_reports_recall() -> None:
    global_scores = torch.tensor([[1.0, 1.0, 0.1], [0.1, 0.2, 0.3]])
    assert stable_global_top_n(global_scores, 2).tolist() == [[0, 1], [2, 1]]
    reranked = torch.tensor([[0.0, 2.0, 100.0], [9.0, 2.0, 3.0]])
    staged = apply_two_stage_reranking(global_scores, reranked, 2)
    assert torch.isneginf(staged[0, 2])
    positives = torch.tensor([[False, True, False], [True, False, False]])
    assert candidate_recall_at_n(global_scores, positives, 2).item() == 0.5


def test_finite_forward_backward_and_checkpoint_round_trip(tmp_path) -> None:
    model, inputs = _inputs()
    output = model(**inputs.as_kwargs())
    loss = output.reranked_score.mean() + output.decoded_mask_logits.sigmoid().mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    path = tmp_path / "v3.pt"
    torch.save(model.state_dict(), path)
    restored, _ = _inputs()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    expected = model.eval()(**inputs.as_kwargs()).reranked_score
    actual = restored.eval()(**inputs.as_kwargs()).reranked_score
    assert torch.equal(expected, actual)
