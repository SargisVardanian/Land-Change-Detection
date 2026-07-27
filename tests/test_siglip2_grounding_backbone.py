from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from land_change_detection.backbones.siglip2_grounding import (
    FrozenSigLIP2GroundingBackbone,
)


class _FakeTokenizer:
    pad_token_id = 0
    all_special_ids = [0, 1]

    def __call__(self, captions, **kwargs):
        rows = []
        for caption in captions:
            direction = 11 if "appeared" in caption else 12
            rows.append([2, direction, 1] + [0] * 61)
        ids = torch.tensor(rows)
        return {"input_ids": ids, "attention_mask": ids.ne(0)}

    def convert_ids_to_tokens(self, ids):
        names = {0: "<pad>", 1: "</s>", 2: "▁new", 11: "▁appeared", 12: "▁demolished"}
        return [names.get(int(value), "▁token") for value in ids]


class _FakeVision(nn.Module):
    def forward(self, *, pixel_values):
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.ones(batch, 256, 768, device=pixel_values.device)
        )


class _FakeText(nn.Module):
    def forward(self, *, input_ids, attention_mask):
        batch, length = input_ids.shape
        values = input_ids.float().unsqueeze(-1).expand(batch, length, 768)
        return SimpleNamespace(last_hidden_state=values)


class _FakeSigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.unregistered_weight_probe = nn.Parameter(torch.ones(1))
        self.vision_model = _FakeVision()
        self.text_model = _FakeText()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=768, image_size=256, patch_size=16),
            text_config=SimpleNamespace(max_position_embeddings=64),
        )


def _install_fake_transformers(monkeypatch) -> None:
    module = ModuleType("transformers")
    module.AutoModel = SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeSigLIP())
    module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer())
    module.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            image_processor=SimpleNamespace(
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_siglip2_dense_contract_and_direction_tokens(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    backbone = FrozenSigLIP2GroundingBackbone("/immutable/siglip2")
    images = backbone.encode_images(torch.rand(2, 2, 3, 256, 256))
    text = backbone.encode_texts(
        ["new buildings appeared", "buildings were demolished"],
        device=torch.device("cpu"),
    )
    assert images.shape == (2, 2, 256, 768)
    assert text.token_embeddings.shape == (2, 64, 768)
    assert text.content_mask[0, 1]
    assert text.content_mask[1, 1]
    assert not torch.equal(text.token_embeddings[0], text.token_embeddings[1])


def test_siglip2_frozen_weights_are_absent_from_optimizer_and_checkpoint(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    backbone = FrozenSigLIP2GroundingBackbone("/immutable/siglip2")
    assert list(backbone.parameters()) == []
    assert backbone.state_dict() == {}
    provenance = backbone.provenance()
    assert provenance["checkpoint_serialization"] == "external_reference_not_embedded"
    assert provenance["dense_grid"] == [16, 16]
