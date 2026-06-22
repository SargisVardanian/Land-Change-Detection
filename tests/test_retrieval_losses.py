from __future__ import annotations

import torch

from land_change_detection.losses.retrieval_losses import (
    asymmetric_caption_pair_loss,
    soft_histogram_contrastive_loss,
    supervised_contrastive_loss,
    symmetric_infonce_loss,
)


def test_symmetric_infonce_loss_runs():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss = symmetric_infonce_loss(embeddings, embeddings)
    assert loss.item() >= 0.0


def test_symmetric_infonce_supports_multi_positive_mask():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    positive_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )
    loss = symmetric_infonce_loss(image_embeddings, text_embeddings, positive_mask=positive_mask)
    assert loss.item() >= 0.0


def test_supervised_contrastive_loss_runs():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=torch.float32)
    loss = supervised_contrastive_loss(embeddings, ["a", "a", "b"])
    assert loss.item() >= 0.0


def test_soft_histogram_contrastive_loss_runs():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=torch.float32)
    histograms = torch.tensor([[0.7, 0.3], [0.65, 0.35], [0.1, 0.9]], dtype=torch.float32)
    loss = soft_histogram_contrastive_loss(embeddings, histograms)
    assert loss.item() >= 0.0


def test_asymmetric_caption_pair_loss_uses_logsumexp_for_pair_to_text():
    pair_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    text_embeddings = torch.tensor(
        [
            [4.0, 0.0],
            [1.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    caption_to_pair = torch.tensor([0, 0, 1], dtype=torch.long)

    loss, stats = asymmetric_caption_pair_loss(pair_embeddings, text_embeddings, caption_to_pair, temperature=1.0)

    logits = text_embeddings @ pair_embeddings.transpose(0, 1)
    expected_pair0 = -(
        torch.logsumexp(torch.tensor([logits[0, 0], logits[1, 0]]), dim=0) - torch.logsumexp(logits[:, 0], dim=0)
    )
    expected_pair1 = -(logits[2, 1] - torch.logsumexp(logits[:, 1], dim=0))
    expected_pair_to_text = torch.stack([expected_pair0, expected_pair1]).mean()

    assert torch.isclose(torch.tensor(stats["pair_to_text_loss"]), expected_pair_to_text, atol=1e-6)
    assert loss.item() >= 0.0
