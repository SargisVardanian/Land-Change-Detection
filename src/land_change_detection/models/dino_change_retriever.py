from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DINOV2_PATH_ERROR = (
    "DINOv2 model path not found. Download it first with scripts/download_dinov2_small.py "
    "or use --visual-backbone simple_patch."
)
REMOTECLIP_PATH_ERROR = (
    "RemoteCLIP model path not found. Download it first through the baseline bundle "
    "or use --text-backbone simple_text."
)
OPENCLIP_IMPORT_ERROR = "open_clip is not installed. Use --text-backbone simple_text or install open_clip_torch."


@dataclass(frozen=True)
class DINOChangeRetrieverConfig:
    visual_backbone: str = "simple_patch"
    text_backbone: str = "remoteclip"
    pair_feature_mode: str = "change_fusion"
    dinov2_model_path: str | None = None
    remoteclip_model_path: str | None = None
    openclip_model_name: str = "ViT-B-32"
    openclip_pretrained: str | None = None
    local_files_only: bool = False
    image_size: int = 224
    patch_size: int = 14
    hidden_dim: int = 384
    transformer_layers: int = 2
    transformer_heads: int = 6
    dropout: float = 0.1


class SimplePatchEncoder(nn.Module):
    def __init__(self, patch_size: int = 14, hidden_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, hidden_dim)
        self.hidden_size = hidden_dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        patch_tokens = patches.transpose(1, 2)
        patch_tokens = self.proj(patch_tokens)
        cls_token = patch_tokens.mean(dim=1)
        return patch_tokens, cls_token


class FrozenDinoV2Encoder(nn.Module):
    def __init__(self, model_path: str | Path, local_files_only: bool = True):
        super().__init__()
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(DINOV2_PATH_ERROR)
        from transformers import AutoImageProcessor, AutoModel

        self.image_processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=local_files_only)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = getattr(self.image_processor, "image_mean", None)
        std = getattr(self.image_processor, "image_std", None)
        if mean is None or std is None:
            return images
        mean_tensor = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        return (images - mean_tensor) / std_tensor.clamp_min(1e-6)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=self._normalize(images))
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0]
        patch_tokens = hidden[:, 1:]
        return patch_tokens, cls_token


class FrozenSimpleTextEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.output_dim = feature_dim

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        features = torch.zeros((len(texts), self.output_dim), dtype=torch.float32, device=device)
        for row_index, text in enumerate(texts):
            for token_index, token in enumerate(text.lower().split()):
                slot = (sum(ord(ch) for ch in token) + token_index) % self.output_dim
                features[row_index, slot] += 1.0
        return F.normalize(features, dim=-1)


class FrozenHFRemoteTextEncoder(nn.Module):
    def __init__(self, model_path: str | Path, local_files_only: bool = True):
        super().__init__()
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(REMOTECLIP_PATH_ERROR)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=local_files_only)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        projection_dim = getattr(self.model.config, "projection_dim", None)
        text_hidden = getattr(getattr(self.model.config, "text_config", None), "hidden_size", None)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.output_dim = int(projection_dim or text_hidden or hidden_size)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if hasattr(self.model, "get_text_features"):
            features = self.model.get_text_features(**encoded)
        else:
            outputs = self.model(**encoded)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            projection = getattr(self.model, "text_projection", None)
            features = pooled if projection is None else projection(pooled)
        return F.normalize(features, dim=-1)


class FrozenOpenClipTextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str | None):
        super().__init__()
        try:
            import open_clip
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(OPENCLIP_IMPORT_ERROR) from exc
        self._open_clip = open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.output_dim = int(getattr(self.model, "text_projection").shape[-1])

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(device)
        features = self.model.encode_text(tokens)
        return F.normalize(features, dim=-1)


class DINOChangeRetriever(nn.Module):
    def __init__(self, config: DINOChangeRetrieverConfig | None = None):
        super().__init__()
        self.config = config or DINOChangeRetrieverConfig()
        if self.config.visual_backbone == "simple_patch":
            self.visual_encoder: nn.Module = SimplePatchEncoder(
                patch_size=self.config.patch_size,
                hidden_dim=self.config.hidden_dim,
            )
            encoder_dim = self.visual_encoder.hidden_size
        elif self.config.visual_backbone == "dinov2":
            model_path = self.config.dinov2_model_path
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(DINOV2_PATH_ERROR)
            self.visual_encoder = FrozenDinoV2Encoder(model_path, local_files_only=self.config.local_files_only)
            encoder_dim = self.visual_encoder.hidden_size
        else:
            raise ValueError(f"Unsupported visual_backbone={self.config.visual_backbone}")

        self._text_encoder: nn.Module | None = None
        self._text_encoder_dim: int | None = None
        self.before_token_type = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        self.after_token_type = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        self.temporal_order_embedding = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        self.change_cls = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        pair_feature_dim = {
            "t2_only": encoder_dim,
            "signed_delta": encoder_dim * 2,
            "change_fusion": encoder_dim * 5,
        }.get(self.config.pair_feature_mode)
        if pair_feature_dim is None:
            raise ValueError(f"Unsupported pair_feature_mode={self.config.pair_feature_mode}")
        self.input_projection = nn.Linear(pair_feature_dim, self.config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.transformer_layers)
        self.text_projection: nn.Linear | None = None
        if self.config.text_backbone == "simple_text":
            self._load_text_encoder()
        elif self.config.text_backbone == "remoteclip" and self.config.remoteclip_model_path and Path(self.config.remoteclip_model_path).exists():
            self._load_text_encoder()
        elif self.config.text_backbone == "openclip" and self.config.openclip_pretrained:
            self._load_text_encoder()

    def _load_text_encoder(self) -> nn.Module:
        if self._text_encoder is not None:
            return self._text_encoder
        if self.config.text_backbone == "simple_text":
            encoder = FrozenSimpleTextEncoder()
        elif self.config.text_backbone == "remoteclip":
            model_path = self.config.remoteclip_model_path
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(REMOTECLIP_PATH_ERROR)
            encoder = FrozenHFRemoteTextEncoder(model_path, local_files_only=self.config.local_files_only)
        elif self.config.text_backbone == "openclip":
            encoder = FrozenOpenClipTextEncoder(self.config.openclip_model_name, self.config.openclip_pretrained)
        else:
            raise ValueError(f"Unsupported text_backbone={self.config.text_backbone}")
        self._text_encoder = encoder
        self._text_encoder_dim = int(getattr(encoder, "output_dim"))
        self.text_projection = nn.Linear(self._text_encoder_dim, self.config.hidden_dim)
        return encoder

    def _encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoder = self._load_text_encoder()
        raw_features = encoder(texts, device)
        assert self.text_projection is not None
        return F.normalize(self.text_projection(F.normalize(raw_features, dim=-1)), dim=-1)

    def forward(self, before: torch.Tensor, after: torch.Tensor, texts: list[str] | None = None) -> dict[str, torch.Tensor]:
        patch_before, cls_before = self.visual_encoder(before)
        patch_after, cls_after = self.visual_encoder(after)
        signed_delta = patch_after - patch_before
        if self.config.pair_feature_mode == "t2_only":
            fused = patch_after
        elif self.config.pair_feature_mode == "signed_delta":
            fused = torch.cat([signed_delta, torch.abs(signed_delta)], dim=-1)
        else:
            fused = torch.cat(
                [
                    patch_before,
                    patch_after,
                    signed_delta,
                    torch.abs(signed_delta),
                    patch_before * patch_after,
                ],
                dim=-1,
            )
        tokens = self.input_projection(fused)
        tokens = tokens + self.before_token_type + self.after_token_type + self.temporal_order_embedding
        change_cls = self.change_cls.expand(before.shape[0], -1, -1)
        encoded = self.transformer(torch.cat([change_cls, tokens], dim=1))
        z_change = F.normalize(encoded[:, 0], dim=-1)
        output = {
            "change_embedding": z_change,
            "before_cls": cls_before,
            "after_cls": cls_after,
            "patch_tokens": encoded[:, 1:],
        }
        if texts is not None:
            output["text_embedding"] = self._encode_text(texts, before.device)
        return output
