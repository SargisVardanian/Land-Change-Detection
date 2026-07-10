from __future__ import annotations

import torch

from land_change_detection.models.qcpr import QCPRPatchReranker, query_segmentation_loss


def test_qcpr_fused_scores_and_mask_shapes():
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8, alpha=1.0, beta=0.25)
    changes = torch.randn(2, 16, 8, requires_grad=True)
    queries = torch.randn(3, 8, requires_grad=True)
    pairs = torch.randn(2, 8, requires_grad=True)

    patches = reranker.project_patches(changes)
    output = reranker.score(queries, pairs, patches)

    assert output["global_score"].shape == (3, 2)
    assert output["local_score"].shape == (3, 2)
    assert output["final_score"].shape == (3, 2)
    assert output["query_mask_logits"].shape == (3, 2, 16)
    assert output["score_mode"] == "fused"
    assert torch.allclose(
        output["final_score"],
        output["global_score"] + 0.25 * output["local_score"],
    )


def test_query_segmentation_loss_is_finite_and_backpropagates_to_both_heads():
    reranker = QCPRPatchReranker(hidden_dim=8, retrieval_dim=8)
    changes = torch.randn(2, 16, 8)
    queries = torch.randn(3, 8)
    pairs = torch.randn(2, 8)
    output = reranker.score(queries, pairs, reranker.project_patches(changes))
    masks = torch.zeros(2, 32, 32)
    masks[0, 4:16, 5:20] = 1.0
    masks[1, 18:28, 18:28] = 1.0

    loss = query_segmentation_loss(
        output["query_mask_logits"],
        torch.tensor([0, 1, 0]),
        masks,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in reranker.patch_projector.parameters())
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in reranker.query_mask_head.parameters())


def test_query_segmentation_loss_rejects_non_square_patch_grid():
    try:
        query_segmentation_loss(torch.randn(1, 1, 15), torch.tensor([0]), torch.zeros(1, 8, 8))
    except ValueError as exc:
        assert "square grid" in str(exc)
    else:
        raise AssertionError("Expected non-square patch grid to be rejected")
