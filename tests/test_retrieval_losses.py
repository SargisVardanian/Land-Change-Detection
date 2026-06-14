from __future__ import annotations

import torch

from land_change_detection.losses.retrieval_losses import (
    soft_histogram_contrastive_loss,
    supervised_contrastive_loss,
    symmetric_infonce_loss,
)


def test_symmetric_infonce_loss_runs():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss = symmetric_infonce_loss(embeddings, embeddings)
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
