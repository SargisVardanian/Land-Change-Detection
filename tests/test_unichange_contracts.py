from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from land_change_detection.backbones.jina_v5_text import _last_token_pool, role_prefix
from land_change_detection.backbones.universat_backend import (
    LevirRGBSpec,
    UniverSatBackendConfig,
    UniverSatJointBackend,
)
from land_change_detection.losses.unichange_losses import masked_multi_positive_sigmoid_loss
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel


class FakeUniverSat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, output_grid: int = 36, **_: object) -> torch.Tensor:
        self.seen_shape = tuple(x.shape)
        return torch.ones(x.shape[0], output_grid * output_grid, 768, device=x.device) * self.weight


def test_text_role_prefixes_and_last_token_pooling() -> None:
    assert role_prefix("query") == "Query: "
    assert role_prefix("document") == "Document: "
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled = _last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_universat_joint_backend_uses_single_temporal_input(tmp_path: Path) -> None:
    fake = FakeUniverSat()
    backend = UniverSatJointBackend(
        UniverSatBackendConfig(source_dir=tmp_path, checkpoint_dir=tmp_path),
        model=fake,
    )
    t1 = torch.rand(2, 3, 64, 64)
    t2 = torch.rand(2, 3, 64, 64)
    features = backend(t1, t2)
    assert fake.seen_shape == (2, 2, 3, 64, 64)
    assert features.local_tokens.shape == (2, 1296, 512)
    assert features.global_embedding.shape == (2, 512)
    assert features.metadata["backend"] == "universat_joint_public"
    assert features.metadata["sensor_spec"]["gsd"]["status"] == "unknown"


def test_unichange_event_decoder_contract(tmp_path: Path) -> None:
    backend = UniverSatJointBackend(
        UniverSatBackendConfig(source_dir=tmp_path, checkpoint_dir=tmp_path),
        model=FakeUniverSat(),
    )
    model = UniChangeModel(visual_encoder=backend, config=UniChangeConfig(event_queries=8))
    out = model(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert out.global_pair_embedding.shape == (2, 512)
    assert out.local_change_tokens.shape == (2, 1296, 512)
    assert out.event_embeddings is not None and out.event_embeddings.shape == (2, 8, 512)
    assert out.event_presence is not None and out.event_presence.shape == (2, 8)
    assert out.event_masks is not None and out.event_masks.shape == (2, 8, 1296)
    assert out.semantic_prediction is not None and out.semantic_prediction.shape == (2, 512)


def test_masked_multi_positive_loss_ignores_unknowns() -> None:
    pairs = torch.nn.functional.normalize(torch.eye(3, 4), dim=-1)
    texts = pairs[[0, 0, 2]]
    caption_to_pair = torch.tensor([0, 0, 2])
    safe_negatives = torch.tensor(
        [
            [False, True, False],
            [False, True, False],
            [True, False, False],
        ]
    )
    loss, stats = masked_multi_positive_sigmoid_loss(pairs, texts, caption_to_pair, safe_negatives)
    assert torch.isfinite(loss)
    assert stats["positive_ratio"] > 0
    assert stats["negative_ratio"] > 0
    assert stats["ignored_ratio"] > 0


def test_levir_rgb_spec_does_not_invent_sensor_dates() -> None:
    metadata = LevirRGBSpec().to_universat_metadata()
    assert metadata["calendar_dates"] is None
    assert metadata["gsd"]["status"] == "unknown"
    assert metadata["sensor_name"].startswith("unknown")
