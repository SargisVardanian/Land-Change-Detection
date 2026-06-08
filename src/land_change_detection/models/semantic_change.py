from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from land_change_detection.semantic_surface import SURFACE_CLASS_DEFS


@dataclass(frozen=True)
class SemanticChangeModelConfig:
    """Trainable semantic-first change model configuration.

    The default channel order is the Prithvi/Sentinel-style six-band stack:
    blue, green, red, narrow NIR, SWIR1, SWIR2.
    """

    input_channels: int = 6
    num_classes: int = len(SURFACE_CLASS_DEFS)
    base_channels: int = 32
    encoder_depth: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class SemanticChangeOutput:
    before_logits: torch.Tensor
    after_logits: torch.Tensor
    change_logits: torch.Tensor

    @property
    def before_semantic_map(self) -> torch.Tensor:
        return self.before_logits.argmax(dim=1)

    @property
    def after_semantic_map(self) -> torch.Tensor:
        return self.after_logits.argmax(dim=1)

    @property
    def change_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.change_logits)

    @property
    def binary_change_map(self) -> torch.Tensor:
        return (self.change_probability >= 0.5).to(torch.uint8).squeeze(1)

    def transition_map(self) -> torch.Tensor:
        return encode_transition_map(self.before_semantic_map, self.after_semantic_map, self.before_logits.shape[1])


def encode_transition_map(before_map: torch.Tensor, after_map: torch.Tensor, num_classes: int) -> torch.Tensor:
    if before_map.shape != after_map.shape:
        raise ValueError(f"semantic maps must match, got {tuple(before_map.shape)} vs {tuple(after_map.shape)}")
    return before_map.to(torch.int64) * int(num_classes) + after_map.to(torch.int64)


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SharedUNetSegmenter(nn.Module):
    def __init__(self, config: SemanticChangeModelConfig):
        super().__init__()
        if config.encoder_depth < 2:
            raise ValueError("encoder_depth must be at least 2")
        channels = [config.base_channels * (2**idx) for idx in range(config.encoder_depth)]
        self.stem = _ConvBlock(config.input_channels, channels[0], config.dropout)
        self.down_blocks = nn.ModuleList(
            _ConvBlock(channels[idx - 1], channels[idx], config.dropout) for idx in range(1, config.encoder_depth)
        )
        self.up_blocks = nn.ModuleList(
            _ConvBlock(channels[idx] + channels[idx - 1], channels[idx - 1], config.dropout)
            for idx in range(config.encoder_depth - 1, 0, -1)
        )
        self.classifier = nn.Conv2d(channels[0], config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips: list[torch.Tensor] = []
        features = self.stem(x)
        skips.append(features)
        for block in self.down_blocks:
            features = F.avg_pool2d(features, kernel_size=2, stride=2)
            features = block(features)
            skips.append(features)

        decoded = skips[-1]
        for block, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = block(torch.cat([decoded, skip], dim=1))

        return self.classifier(decoded), decoded


class SemanticChangeModel(nn.Module):
    """Semantic-first bi-temporal model.

    The shared segmenter predicts land-cover classes for T1 and T2 separately.
    A small auxiliary change head then uses the decoded T1/T2 features and
    their absolute difference to predict changed pixels. The semantic maps are
    the primary output; the binary map is for localization QA and benchmarking.
    """

    def __init__(self, config: SemanticChangeModelConfig | None = None):
        super().__init__()
        self.config = config or SemanticChangeModelConfig()
        self.segmenter = _SharedUNetSegmenter(self.config)
        feature_channels = self.config.base_channels
        self.change_head = nn.Sequential(
            _ConvBlock(feature_channels * 3, feature_channels, self.config.dropout),
            nn.Conv2d(feature_channels, 1, kernel_size=1),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> SemanticChangeOutput:
        self._validate_pair(before, after)
        before_logits, before_features = self.segmenter(before)
        after_logits, after_features = self.segmenter(after)
        change_features = torch.cat([before_features, after_features, torch.abs(after_features - before_features)], dim=1)
        change_logits = self.change_head(change_features)
        return SemanticChangeOutput(
            before_logits=before_logits,
            after_logits=after_logits,
            change_logits=change_logits,
        )

    def _validate_pair(self, before: torch.Tensor, after: torch.Tensor) -> None:
        if before.shape != after.shape:
            raise ValueError(f"before/after tensors must match, got {tuple(before.shape)} vs {tuple(after.shape)}")
        if before.ndim != 4:
            raise ValueError(f"expected NCHW tensors, got shape={tuple(before.shape)}")
        if before.shape[1] != self.config.input_channels:
            raise ValueError(f"expected {self.config.input_channels} input channels, got {before.shape[1]}")


def build_semantic_change_model(config: SemanticChangeModelConfig | None = None) -> SemanticChangeModel:
    return SemanticChangeModel(config=config)
