from types import SimpleNamespace

import torch

from land_change_detection.models.qcpr_v3_explanations import (
    apply_selected_counterfactual,
    contribution_mass_selection,
    counterfactual_evidence,
    directional_counterfactuals,
    reconstruct_final_score,
)


def test_directional_counterfactuals_appeared_changes_only_t2_tile():
    images = torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4).float()
    variants, boxes = directional_counterfactuals(images, grid=2, direction="appeared")
    y0, y1, x0, x1 = boxes[0]
    expected = images.clone(); expected[1, :, y0:y1, x0:x1] = images[0, :, y0:y1, x0:x1]
    torch.testing.assert_close(variants[0], expected)
    torch.testing.assert_close(variants[0, 0], images[0])


def test_directional_counterfactuals_disappeared_changes_only_t1_tile():
    images = torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4).float()
    variants, boxes = directional_counterfactuals(images, grid=2, direction="disappeared")
    y0, y1, x0, x1 = boxes[-1]
    expected = images.clone(); expected[0, :, y0:y1, x0:x1] = images[1, :, y0:y1, x0:x1]
    torch.testing.assert_close(variants[-1], expected)
    torch.testing.assert_close(variants[-1, 1], images[1])


def test_selected_counterfactual_preserves_everything_outside_selected_tile():
    images = torch.randn(2, 3, 5, 7)
    _, boxes = directional_counterfactuals(images, grid=2, direction="appeared")
    selected = torch.tensor([[False, True], [False, False]])
    result = apply_selected_counterfactual(images, boxes, selected, direction="appeared")
    y0, y1, x0, x1 = boxes[1]
    expected = images.clone(); expected[1, :, y0:y1, x0:x1] = images[0, :, y0:y1, x0:x1]
    torch.testing.assert_close(result, expected)


def test_color_matched_counterfactual_keeps_destination_tile_mean():
    images = torch.zeros(2, 3, 4, 4)
    images[0] = 2.0
    images[1] = 10.0
    variants, boxes = directional_counterfactuals(images, grid=2, direction="appeared", replacement="color_matched")
    y0, y1, x0, x1 = boxes[0]
    torch.testing.assert_close(variants[0, 1, :, y0:y1, x0:x1].mean(), images[1, :, y0:y1, x0:x1].mean())
    torch.testing.assert_close(variants[0, 1, :, y1:, x1:], images[1, :, y1:, x1:])


def test_contribution_mass_selects_minimum_tiles_and_empty_has_no_fallback():
    selection = contribution_mass_selection(torch.tensor([[6.0, 3.0], [1.0, -9.0]]), mass=0.8)
    assert selection.selected_count == 2
    assert selection.mask.tolist() == [[True, True], [False, False]]
    empty = contribution_mass_selection(torch.zeros(2, 2))
    assert not empty.mask.any()
    assert empty.status == "NO_FAITHFUL_SPATIAL_EVIDENCE"


def test_score_reconstruction_matches_reranked_score():
    logits = torch.tensor([-1.0, 0.5])
    global_score = torch.tensor([[0.2]])
    local = torch.tensor([[0.3]])
    token = torch.tensor([[0.4]])
    reranked = global_score + torch.nn.functional.softplus(logits[0]) * local + torch.nn.functional.softplus(logits[1]) * token
    scores = SimpleNamespace(global_score=global_score, local_score=local, token_patch_score=token, reranked_score=reranked)
    torch.testing.assert_close(reconstruct_final_score(scores, logits), reranked)


def test_chunked_counterfactual_scoring_matches_unchunked_and_localizes_toy_tile():
    images = torch.zeros(2, 1, 4, 4); images[1, :, :2, :2] = 5
    def score(batch):
        return batch[:, 1, :, :2, :2].mean((1, 2, 3))
    original_a, evidence_a, _ = counterfactual_evidence(score, images, grid=2, direction="appeared", chunk_size=1)
    original_b, evidence_b, _ = counterfactual_evidence(score, images, grid=2, direction="appeared", chunk_size=4)
    torch.testing.assert_close(original_a, original_b)
    torch.testing.assert_close(evidence_a, evidence_b)
    assert evidence_a.argmax().item() == 0
    assert evidence_a[0, 0] > 0 and torch.count_nonzero(evidence_a) == 1
