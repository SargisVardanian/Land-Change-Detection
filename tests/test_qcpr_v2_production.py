from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from land_change_detection.backbones.jina_v5_text import TextFeatures
from land_change_detection.models.qcpr import QCPRPatchReranker, local_positive_negative_margin_loss, temporal_channel_loss_components
from land_change_detection.models.unichange_v2_retrieval import UniChangeV2RetrievalModel
from ucv2_retrieval_metrics import global_top_n_rerank_scores
from ucv2_stage1_next_core import Stage1NextConfig, freeze_global_retrieval_modules, make_optimizer


def test_qcpr_v1_state_dict_strict_roundtrip_and_legacy_score() -> None:
    first = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v1")
    second = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v1")
    second.load_state_dict(first.state_dict(), strict=True)
    output = second.score(torch.randn(2, 8), torch.randn(3, 8), second.project_patches(torch.randn(3, 4, 8)))
    assert output["score_mode"] == "fused"
    assert torch.allclose(output["local_score"], output["query_mask_logits"].sigmoid().amax(-1))


def test_qcpr_v2_token_conditioned_shapes_and_gradients() -> None:
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    output = reranker.score_v2(torch.randn(2, 8), torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool), torch.randn(3, 8), torch.randn(3, 2, 4, 8))
    assert output["query_mask_logits"].shape == (2, 3, 4)
    assert output["temporal_explanation_logits"].shape == (3, 4, 3)
    assert torch.allclose(output["final_score"], output["S_final"])
    assert torch.allclose(output["S_token_patch_raw"], output["token_patch_score"])
    assert not torch.allclose(
        output["final_score"],
        output["fusion_weights"][0] * output["S_global_calibrated"]
        + output["fusion_weights"][1] * output["S_local_calibrated"],
    )
    assert torch.all(output["fusion_weights"] > 0)
    assert torch.allclose(output["fusion_weights"].sum(), torch.tensor(1.0))
    output["final_score"].sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in reranker.interaction_mlp.parameters())
    assert reranker.fusion_logits.grad is not None and torch.any(reranker.fusion_logits.grad != 0)
    assert reranker.branch_log_scales.grad is not None and torch.any(reranker.branch_log_scales.grad != 0)
    assert "branch_biases" not in dict(reranker.named_parameters())
    assert "branch_biases" in dict(reranker.named_buffers())


def test_v2_global_score_matches_v1_compatible_initial_global_retrieval() -> None:
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    query = torch.randn(2, 8)
    pairs = torch.randn(3, 8)
    output = reranker.score_v2(query, torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool), pairs, torch.randn(3, 2, 4, 8))
    expected = torch.nn.functional.normalize(query, dim=-1) @ torch.nn.functional.normalize(pairs, dim=-1).T
    assert torch.allclose(output["global_score"], expected)


def test_qcpr_v2_query_mask_uses_changed_channel_as_temporal_prior() -> None:
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    with torch.no_grad():
        for parameter in reranker.interaction_mlp.parameters():
            parameter.zero_()
        for parameter in reranker.temporal_channel_head.parameters():
            parameter.zero_()
        reranker.temporal_channel_head[-1].bias[0] = 2.0
    output = reranker.score_v2(
        torch.randn(2, 8), torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool),
        torch.randn(3, 8), torch.randn(3, 2, 4, 8),
    )
    assert torch.allclose(output["query_mask_logits"], torch.full((2, 3, 4), 2.0))


def test_qcpr_v2_model_forward_does_not_require_legacy_mask_query_embeddings() -> None:
    class Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = nn.Identity()

        def forward(self, images):
            return SimpleNamespace(features=images, metadata={})

    class Temporal(nn.Module):
        def forward(self, features, temporal_valid_mask=None):
            batch = features.shape[0]
            return SimpleNamespace(pair_embedding=torch.randn(batch, 8), per_time_tokens=torch.randn(batch, 2, 4, 8))

    class Text(nn.Module):
        def forward(self, captions, role):
            count = len(captions)
            return TextFeatures(torch.randn(count, 8), torch.randn(count, 5, 8), torch.ones(count, 5, dtype=torch.bool), role, {})

    class Retrieval(nn.Module):
        def forward(self, embedding):
            return SimpleNamespace(pair_embedding=embedding)

    model = UniChangeV2RetrievalModel(
        Visual(), Temporal(), Text(), Retrieval(),
        patch_reranker=QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2"),
    )
    output = model(torch.randn(3, 2, 4, 8), ["one", "two"], torch.tensor([0, 1]))
    assert output.score_mode == "qcpr_v2"
    assert output.mask_query_embeddings is None
    assert output.query_mask_logits.shape == (2, 3, 4)


def test_qcpr_v2_token_patch_aggregation_is_not_a_single_maximum() -> None:
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    query_tokens = torch.randn(1, 4, 8)
    attention = torch.tensor([[True, True, True, False]])
    per_time = torch.randn(1, 2, 16, 8)
    output = reranker.score_v2(
        torch.randn(1, 8),
        query_tokens,
        attention,
        torch.randn(1, 8),
        per_time,
    )
    projected_tokens = torch.nn.functional.normalize(reranker.token_projection(query_tokens), dim=-1)
    similarities = torch.einsum("bnd,qld->qbnl", output["patch_tokens"], projected_tokens)
    evidence = similarities.masked_fill(~attention[:, None, None, :], -1.0).amax(dim=-1)
    expected = evidence.topk(4, dim=-1).values.mean(dim=-1)
    assert torch.allclose(output["token_patch_score"], expected)
    assert not torch.allclose(output["token_patch_score"], output["query_mask_logits"].sigmoid().amax(dim=-1))


def test_qcpr_v2_reports_token_group_grounding_scores() -> None:
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    attention = torch.ones(2, 6, dtype=torch.bool)
    group_masks = {name: torch.zeros_like(attention) for name in ("object", "direction", "location", "count", "relation")}
    for index, name in enumerate(group_masks):
        group_masks[name][:, index + 1] = True
    output = reranker.score_v2(
        torch.randn(2, 8), torch.randn(2, 6, 8), attention, torch.randn(3, 8), torch.randn(3, 2, 16, 8),
        query_metadata={
            "token_group_masks": group_masks,
            "direction_targets": torch.tensor([1, -1]),
            "location_targets": torch.tensor([[0.5, 0.0], [1.0, 0.5]]),
            "count_targets": torch.tensor([2.0, 3.0]),
        },
    )
    for name in ("S_object", "S_direction", "S_location", "S_count", "S_relation"):
        assert output[name].shape == (2, 3)
        assert torch.isfinite(output[name]).all()


def test_direction_losses_use_real_supervision_and_reversal_swap() -> None:
    logits = torch.randn(3, 4, 3, requires_grad=True)
    reverse = logits.detach().clone()[..., [0, 2, 1]].requires_grad_()
    result = temporal_channel_loss_components(logits, torch.ones(3, 8, 8), torch.ones(3, 8, 8), ["query_specific", "binary_generic", "none"], torch.tensor([1.0, 0.5, 0.0]), ["appeared", "disappeared", None], reverse)
    assert result["temporal_supervised_pairs"] == 2
    assert result["temporal_reversal_consistency_loss"].item() == 0.0
    total = result["changed_channel_loss"] + result["appeared_channel_loss"] + result["disappeared_channel_loss"]
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_v2_parameters_are_in_optimizer_exactly_once() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_encoder = nn.Linear(8, 8)
            self.retrieval_head = nn.Linear(8, 8)
            self.text_adapter = None
            self.patch_reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, architecture_version="v2")
    model = Model()
    config = Stage1NextConfig("d", "o", "u", "v", "j", qcpr_architecture_version="v2", enable_patch_reranker=True, enable_temporal_explanation_channels=True)
    optimizer = make_optimizer(model, config)
    identities = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(identities) == len(set(identities)) == len([parameter for parameter in model.parameters() if parameter.requires_grad])


def test_two_stage_reranker_cannot_promote_outside_global_candidate_pool() -> None:
    global_scores = torch.tensor([[0.9, 0.8, 0.1]])
    rerank_scores = torch.tensor([[0.1, 0.2, 100.0]])
    output = global_top_n_rerank_scores(global_scores, rerank_scores, top_n=2)
    assert torch.isneginf(output[0, 2])
    assert int(output.argmax(dim=1).item()) == 1


def test_count_margin_loss_is_noop_without_valid_count_negatives() -> None:
    scores = torch.tensor([[0.8, 0.2], [0.7, 0.1]], requires_grad=True)
    captions = ["one building appeared", "one road appeared"]
    loss, diagnostics = local_positive_negative_margin_loss(scores, torch.tensor([0, 1]), captions)
    assert diagnostics["hard_negative_same_object_location_wrong_count_available_queries"] == 0
    assert diagnostics["hard_negative_same_object_location_wrong_count_selected_anchors"] == 0
    assert diagnostics["hard_negative_same_object_location_wrong_count_loss"] == 0.0
    assert torch.isfinite(loss)


def test_local_objective_does_not_update_frozen_global_modules() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal_encoder = nn.Linear(4, 4)
            self.retrieval_head = nn.Linear(4, 4)
            self.text_adapter = nn.Linear(4, 4)
            self.patch_reranker = nn.Linear(4, 1)
    model = Model()
    frozen = freeze_global_retrieval_modules(model)
    assert frozen == ["temporal_encoder", "retrieval_head", "text_adapter"]
    before = [parameter.detach().clone() for module in (model.temporal_encoder, model.retrieval_head, model.text_adapter) for parameter in module.parameters()]
    optimizer = torch.optim.SGD(model.patch_reranker.parameters(), lr=0.1)
    optimizer.zero_grad()
    model.patch_reranker(torch.ones(3, 4)).sum().backward()
    optimizer.step()
    after = [parameter.detach() for module in (model.temporal_encoder, model.retrieval_head, model.text_adapter) for parameter in module.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))
