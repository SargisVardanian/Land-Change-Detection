from __future__ import annotations

import torch
from torch import nn

from land_change_detection.models.qcpr import QCPRPatchReranker, temporal_channel_loss_components
from ucv2_stage1_next_core import Stage1NextConfig, make_optimizer


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
    output["final_score"].sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in reranker.interaction_mlp.parameters())


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
