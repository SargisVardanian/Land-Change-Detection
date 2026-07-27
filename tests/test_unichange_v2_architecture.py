from __future__ import annotations

import torch
from torch import nn

from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
from land_change_detection.data.temporal_collate import collate_temporal_change_samples
from land_change_detection.data.temporal_sample import TemporalChangeSample
from land_change_detection.models.event_decoder import EventDecoder, EventDecoderConfig
from land_change_detection.models.retrieval_heads import multi_positive_symmetric_info_nce, supervised_contrastive_loss
from land_change_detection.models.temporal_change_encoder import TemporalChangeEncoder, TemporalChangeEncoderConfig
from land_change_detection.models.text_conditioned_mask_decoder import TextConditionedMaskDecoder, TextConditionedMaskDecoderConfig
from land_change_detection.training.contrastive_queue import ContrastiveQueue
from land_change_detection.training.task_sampler import retrieval_stage_sampler


class _FakeUniverSat(nn.Module):
    def __init__(self, tokens: int = 16, dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.tokens = tokens
        self.calls: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor, output_grid: int | None = None):
        self.calls.append(tuple(images.shape))
        pooled = images.mean(dim=(-2, -1))
        token = self.proj(pooled)
        return token.unsqueeze(1).expand(images.shape[0], self.tokens, token.shape[-1])


def test_temporal_collate_preserves_variable_t_and_resize_modes():
    samples = [
        TemporalChangeSample(
            images=torch.rand(2, 3, 12, 12),
            timestamps=torch.tensor([0.0, 1.0]),
            pair_id="a",
            captions=["new building"],
            binary_mask=torch.ones(12, 12),
        ),
        TemporalChangeSample(
            images=torch.rand(3, 3, 10, 10),
            timestamps=torch.tensor([0.0, 1.0, 2.0]),
            pair_id="b",
            captions=["road extended"],
            binary_mask=torch.zeros(10, 10),
        ),
    ]

    batch = collate_temporal_change_samples(samples, image_size=16)

    assert batch.images.shape == (2, 3, 3, 16, 16)
    assert batch.timestamps.shape == (2, 3)
    assert batch.temporal_valid_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.binary_mask is not None
    assert set(batch.binary_mask[0].unique().tolist()) == {1.0}
    assert set(batch.binary_mask[1].unique().tolist()) == {0.0}


def test_sequence_universat_flattens_b_times_t_and_restores_temporal_axis():
    fake = _FakeUniverSat(tokens=16, dim=8)
    encoder = SequenceUniverSatEncoder(fake, output_grid=4, visual_dim=8, freeze=True)

    output = encoder(torch.rand(2, 3, 3, 32, 32))

    assert fake.calls == [(6, 3, 32, 32)]
    assert output.features.shape == (2, 3, 16, 8)
    assert all(not parameter.requires_grad for parameter in fake.parameters())


def test_temporal_encoder_event_decoder_and_query_mask_shapes():
    encoder = TemporalChangeEncoder(
        TemporalChangeEncoderConfig(
            input_dim=8,
            hidden_dim=32,
            depth=2,
            heads=4,
            ffn_dim=64,
            grid_size=4,
            window_size=2,
            global_tokens=4,
        )
    )
    features = torch.rand(2, 3, 16, 8)
    valid = torch.tensor([[True, True, False], [True, True, True]])

    encoded = encoder(features, temporal_valid_mask=valid)

    assert encoded.per_time_tokens.shape == (2, 3, 16, 32)
    assert encoded.change_tokens.shape == (2, 16, 32)
    assert encoded.global_tokens.shape == (2, 4, 32)
    assert encoded.pair_embedding.shape == (2, 32)
    assert torch.allclose(encoded.pair_embedding.norm(dim=-1), torch.ones(2), atol=1e-5)

    event_decoder = EventDecoder(EventDecoderConfig(hidden_dim=32, event_queries=3, heads=4, ffn_dim=64, grid_size=4, mask_size=256))
    events = event_decoder(encoded.change_tokens)
    assert events.event_embeddings.shape == (2, 3, 32)
    assert events.event_presence_logits.shape == (2, 3)
    assert events.event_mask_logits.shape == (2, 3, 256, 256)

    mask_decoder = TextConditionedMaskDecoder(TextConditionedMaskDecoderConfig(hidden_dim=32, heads=4, layers=1, grid_size=4, mask_size=256))
    soft_mask = mask_decoder(
        text_tokens=torch.rand(2, 5, 32),
        change_tokens=encoded.change_tokens,
        event_embeddings=events.event_embeddings,
        event_masks=torch.sigmoid(events.event_mask_logits),
        text_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert soft_mask.shape == (2, 256, 256)


def test_retrieval_losses_queue_and_task_sampler_are_stage_separated():
    pair = torch.eye(3, 4)
    text = torch.eye(3, 4)
    caption_to_pair = torch.tensor([0, 1, 2])
    assert torch.isfinite(multi_positive_symmetric_info_nce(pair, text, caption_to_pair))
    assert torch.isfinite(supervised_contrastive_loss(pair, torch.tensor([1, 1, 2])))

    queue = ContrastiveQueue(capacity=2, dim=4)
    queue.enqueue(pair, ["a", "b", "c"])
    snapshot = queue.snapshot()
    assert len(queue) == 2
    assert snapshot.labels == ["b", "c"]

    sampler = retrieval_stage_sampler()
    routes = [sampler.next().name for _ in range(4)]
    assert routes == ["text_to_pair", "text_to_pair", "text_to_pair", "pair_to_pair"]
