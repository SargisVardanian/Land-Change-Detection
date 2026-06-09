from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LevirBinaryRetrievalConfig:
    image_channels: int = 3
    image_feature_dim: int = 16
    text_feature_dim: int = 256
    hidden_dim: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1


class FrozenSimpleTextEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        features = torch.zeros((len(texts), self.feature_dim), dtype=torch.float32)
        for row_index, text in enumerate(texts):
            for token_index, token in enumerate(text.lower().split()):
                slot = (sum(ord(ch) for ch in token) + token_index) % self.feature_dim
                features[row_index, slot] += 1.0
        return F.normalize(features, dim=-1)


class FrozenSimpleImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        blur = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        edge_x = F.conv2d(gray, self.kernel_x, padding=1)
        edge_y = F.conv2d(gray, self.kernel_y, padding=1)
        edge_mag = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
        pooled_rgb = F.avg_pool2d(images, kernel_size=2, stride=2)
        pooled_gray = F.avg_pool2d(gray, kernel_size=2, stride=2)
        pooled_blur = F.avg_pool2d(blur, kernel_size=2, stride=2)
        pooled_edge_x = F.avg_pool2d(edge_x, kernel_size=2, stride=2)
        pooled_edge_y = F.avg_pool2d(edge_y, kernel_size=2, stride=2)
        pooled_edge_mag = F.avg_pool2d(edge_mag, kernel_size=2, stride=2)
        return torch.cat(
            [pooled_rgb, pooled_gray, pooled_blur, pooled_edge_x, pooled_edge_y, pooled_edge_mag],
            dim=1,
        )


class ChangeFusionTransformer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        fused = self.encoder(tokens)
        return fused.transpose(1, 2).reshape(batch, channels, height, width)


class BinarySegmentationDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


class LevirBinaryRetrievalModel(nn.Module):
    def __init__(self, config: LevirBinaryRetrievalConfig | None = None):
        super().__init__()
        self.config = config or LevirBinaryRetrievalConfig()
        self.image_encoder = FrozenSimpleImageEncoder()
        self.text_encoder = FrozenSimpleTextEncoder(self.config.text_feature_dim)
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        image_out_channels = 8
        fusion_in_channels = image_out_channels * 4
        self.visual_projection = nn.Sequential(
            nn.Conv2d(fusion_in_channels, self.config.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.text_projection = nn.Linear(self.config.text_feature_dim, self.config.hidden_dim)
        self.fusion_transformer = ChangeFusionTransformer(
            hidden_dim=self.config.hidden_dim,
            heads=self.config.transformer_heads,
            layers=self.config.transformer_layers,
            dropout=self.config.dropout,
        )
        self.segmentation_decoder = BinarySegmentationDecoder(self.config.hidden_dim)
        self.retrieval_projection = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.text_encoder(texts).to(device=device)
        projected = self.text_projection(encoded)
        return F.normalize(projected, dim=-1)

    def forward(self, before: torch.Tensor, after: torch.Tensor, texts: list[str]) -> dict[str, torch.Tensor]:
        features_before = self.image_encoder(before)
        features_after = self.image_encoder(after)
        fused_inputs = torch.cat(
            [
                features_before,
                features_after,
                torch.abs(features_after - features_before),
                features_before * features_after,
            ],
            dim=1,
        )
        projected = self.visual_projection(fused_inputs)
        fused_spatial = self.fusion_transformer(projected)
        mask_logits = self.segmentation_decoder(fused_spatial)
        pooled_change = fused_spatial.mean(dim=(2, 3))
        change_embedding = F.normalize(self.retrieval_projection(pooled_change), dim=-1)
        text_embedding = self.encode_text(texts, device=before.device)
        return {
            "mask_logits": mask_logits,
            "change_embedding": change_embedding,
            "text_embedding": text_embedding,
        }


def dice_loss_from_logits(mask_logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def symmetric_contrastive_loss(change_embedding: torch.Tensor, text_embedding: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = change_embedding @ text_embedding.transpose(0, 1) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_i + loss_t)


def segmentation_metrics(mask_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(mask_logits)
    preds = (probs >= threshold).to(torch.int64)
    truth = (targets >= 0.5).to(torch.int64)
    tp = float(((preds == 1) & (truth == 1)).sum().item())
    fp = float(((preds == 1) & (truth == 0)).sum().item())
    fn = float(((preds == 0) & (truth == 1)).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def retrieval_metrics(change_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> dict[str, float]:
    similarities = change_embeddings @ text_embeddings.transpose(0, 1)
    ranks: list[int] = []
    for index in range(similarities.shape[0]):
        ranking = torch.argsort(similarities[index], descending=True)
        rank = int((ranking == index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    total = max(len(ranks), 1)
    return {
        "recall@1": sum(1 for rank in ranks if rank <= 1) / total,
        "recall@5": sum(1 for rank in ranks if rank <= 5) / total,
        "recall@10": sum(1 for rank in ranks if rank <= 10) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
    }
