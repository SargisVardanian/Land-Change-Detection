from __future__ import annotations

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
    assert output.patch_mask_logits.shape == (3, 4, 20)
    assert output.decoded_mask_logits.shape == (3, 4, 32, 32)
    assert output.temporal_descriptors.shape == (4, 20, 16)
    assert output.slot_activations is None
    names = set(dict(model.named_modules()))
    assert not any(term in name for name in names for term in ("object_head", "direction_head", "location_head", "count_head", "relation_head"))


def test_all_consumers_share_exact_scores_and_mask_logits() -> None:
    model, inputs = _inputs()
    outputs = [trainer_score(model, inputs), evaluator_score(model, inputs), renderer_score(model, inputs)]
    for output in outputs[1:]:
        assert torch.equal(outputs[0].reranked_score, output.reranked_score)
        assert torch.equal(outputs[0].decoded_mask_logits, output.decoded_mask_logits)
        assert torch.equal(outputs[0].patch_mask_logits, output.patch_mask_logits)
    assert torch.equal(faithful_mask_score(model, inputs), outputs[0].local_score)


def test_multiscale_decoder_keeps_small_local_peak() -> None:
    model, _ = _inputs()
    logits = torch.full((1, 1, 20), -12.0)
    logits[..., 0] = 12.0
    decoded = model.mask_decoder(logits, ((0, 16, 4), (16, 20, 2))).sigmoid()
    assert decoded.max() > 0.5
    assert 0 < (decoded > 0.5).sum() < decoded.numel()


def test_patch_pooling_logits_are_sampled_from_displayed_mask_field() -> None:
    model, inputs = _inputs()
    output = model(**inputs.as_kwargs())
    expected = model.mask_decoder.sample_patch_logits(
        output.decoded_mask_logits,
        ((0, 16, 4), (16, 20, 2)),
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
