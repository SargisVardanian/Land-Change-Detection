from __future__ import annotations

import sys
import types

import pytest
import torch

from land_change_detection.models.dino_change_retriever import (
    DINOV2_PATH_ERROR,
    REMOTECLIP_PATH_ERROR,
    DINOChangeRetriever,
    DINOChangeRetrieverConfig,
)


def test_simple_patch_forward_pass_runs_without_external_weights():
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="simple_text",
            hidden_dim=96,
            transformer_heads=6,
        )
    )
    before = torch.randn(2, 3, 224, 224)
    after = torch.randn(2, 3, 224, 224)
    output = model(before, after, ["new building detected", "road widened in the south"])
    assert output["change_embedding"].shape == (2, 96)
    assert output["text_embedding"].shape == (2, 96)
    assert output["patch_tokens"].ndim == 3


def test_dinov2_missing_path_fails_with_clear_message(tmp_path):
    missing = tmp_path / "dinov2-small"
    with pytest.raises(FileNotFoundError, match="DINOv2 model path not found"):
        DINOChangeRetriever(
            DINOChangeRetrieverConfig(
                visual_backbone="dinov2",
                dinov2_model_path=str(missing),
                local_files_only=True,
            )
    )
    assert "simple_patch" in DINOV2_PATH_ERROR


def test_remoteclip_missing_path_fails_with_clear_message():
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="remoteclip",
            remoteclip_model_path=None,
            hidden_dim=96,
            transformer_heads=6,
        )
    )
    before = torch.randn(1, 3, 224, 224)
    after = torch.randn(1, 3, 224, 224)
    with pytest.raises(FileNotFoundError, match="RemoteCLIP model path not found"):
        model(before, after, ["new building"])
    assert "simple_text" in REMOTECLIP_PATH_ERROR


def test_dinov2_hidden_size_is_inferred_from_loaded_model(monkeypatch, tmp_path):
    class FakeImageProcessor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.25, 0.25, 0.25]

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=72)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def forward(self, pixel_values: torch.Tensor):
            batch = pixel_values.shape[0]
            hidden = torch.ones(batch, 17, 72, dtype=pixel_values.dtype, device=pixel_values.device)
            return types.SimpleNamespace(last_hidden_state=hidden)

    fake_transformers = types.SimpleNamespace(AutoImageProcessor=FakeImageProcessor, AutoModel=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    model_dir = tmp_path / "dinov2-base"
    model_dir.mkdir()

    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="dinov2",
            text_backbone="simple_text",
            dinov2_model_path=str(model_dir),
            hidden_dim=96,
            transformer_heads=6,
            local_files_only=True,
        )
    )

    assert model.input_projection.in_features == 72 * 5


def test_pair_feature_modes_adjust_input_projection():
    model_t2 = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="simple_text",
            pair_feature_mode="t2_only",
            hidden_dim=64,
            transformer_heads=4,
        )
    )
    model_delta = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="simple_text",
            pair_feature_mode="signed_delta",
            hidden_dim=64,
            transformer_heads=4,
        )
    )
    assert model_t2.input_projection.in_features == 64
    assert model_delta.input_projection.in_features == 128


def test_remoteclip_text_encoder_uses_local_hf_components(monkeypatch, tmp_path):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
            del padding, truncation
            max_len = max(len(text.split()) for text in texts)
            input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
            for row_index, text in enumerate(texts):
                tokens = text.split()
                attention_mask[row_index, : len(tokens)] = 1
                input_ids[row_index, : len(tokens)] = torch.arange(1, len(tokens) + 1)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.config = types.SimpleNamespace(projection_dim=32)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            del attention_mask
            features = torch.zeros(input_ids.shape[0], 32, dtype=torch.float32, device=input_ids.device)
            features[:, 0] = input_ids.sum(dim=1).float()
            features[:, 1] = input_ids.shape[1]
            return features * self.scale

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModel=FakeTextModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    model_dir = tmp_path / "remoteclip"
    model_dir.mkdir()

    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="remoteclip",
            remoteclip_model_path=str(model_dir),
            hidden_dim=64,
            transformer_heads=4,
            local_files_only=True,
        )
    )
    before = torch.randn(2, 3, 224, 224)
    after = torch.randn(2, 3, 224, 224)
    output = model(before, after, ["new building appears", "road gets wider"])

    assert output["text_embedding"].shape == (2, 64)
