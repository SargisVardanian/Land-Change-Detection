from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DINOV2_PATH_ERROR = (
    "DINOv2 model path not found. Download it first with scripts/download_dinov2_small.py "
    "or use --visual-backbone simple_patch."
)


@dataclass(frozen=True)
class DINOChangeRetrieverConfig:
    visual_backbone: str = "simple_patch"
    dinov2_model_path: str | None = None
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
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=local_files_only)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=images)
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, 0]
        patch_tokens = hidden[:, 1:]
        return patch_tokens, cls_token


class DINOChangeRetriever(nn.Module):
    def __init__(self, config: DINOChangeRetrieverConfig | None = None):
        super().__init__()
        self.config = config or DINOChangeRetrieverConfig()
        if self.config.visual_backbone == "simple_patch":
            self.visual_encoder: nn.Module = SimplePatchEncoder(
                patch_size=self.config.patch_size,
                hidden_dim=self.config.hidden_dim,
            )
            encoder_dim = self.config.hidden_dim
        elif self.config.visual_backbone == "dinov2":
            model_path = self.config.dinov2_model_path
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(DINOV2_PATH_ERROR)
            self.visual_encoder = FrozenDinoV2Encoder(model_path, local_files_only=self.config.local_files_only)
            encoder_dim = self.config.hidden_dim
        else:
            raise ValueError(f"Unsupported visual_backbone={self.config.visual_backbone}")

        self.change_cls = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        self.input_projection = nn.Linear(encoder_dim * 4, self.config.hidden_dim)
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
        self.text_projection = nn.Linear(256, self.config.hidden_dim)
        self.text_encoder_dim = 256

    def _encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        features = torch.zeros((len(texts), self.text_encoder_dim), dtype=torch.float32, device=device)
        for row_index, text in enumerate(texts):
            for token_index, token in enumerate(text.lower().split()):
                slot = (sum(ord(ch) for ch in token) + token_index) % self.text_encoder_dim
                features[row_index, slot] += 1.0
        return F.normalize(self.text_projection(F.normalize(features, dim=-1)), dim=-1)

    def forward(self, before: torch.Tensor, after: torch.Tensor, texts: list[str] | None = None) -> dict[str, torch.Tensor]:
        patch_before, cls_before = self.visual_encoder(before)
        patch_after, cls_after = self.visual_encoder(after)
        fused = torch.cat([patch_before, patch_after, torch.abs(patch_after - patch_before), patch_before * patch_after], dim=-1)
        tokens = self.input_projection(fused)
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
