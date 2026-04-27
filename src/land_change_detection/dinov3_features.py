from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .contracts import SegmentationArtifact

DINO_V3_FACEBOOK_VITL16_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"
DINO_V3_VITS16 = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_V3_VITS16_DIR = Path("artifacts/models/features/timm-vit_small_patch16_dinov3.lvd1689m")
DINO_V3_VITB16 = "timm/vit_base_patch16_dinov3.lvd1689m"
DINO_V3_VITB16_DIR = Path("artifacts/models/features/timm-vit_base_patch16_dinov3.lvd1689m")
DINO_V3_VITL16_SAT = "timm/vit_large_patch16_dinov3.sat493m"
DINO_V3_VITL16_SAT_DIR = Path("artifacts/models/features/timm-vit_large_patch16_dinov3.sat493m")
DINO_REGION_ID2LABEL = {
    0: "dino_region_1",
    1: "dino_region_2",
    2: "dino_region_3",
    3: "dino_region_4",
    4: "dino_region_5",
    5: "dino_region_6",
}
DINO_REGION_COLORS = {
    "dino_region_1": "#2c3e50",
    "dino_region_2": "#16a085",
    "dino_region_3": "#f39c12",
    "dino_region_4": "#8e44ad",
    "dino_region_5": "#c0392b",
    "dino_region_6": "#7f8c8d",
}


@dataclass(frozen=True)
class DINOCellShift:
    cell: str
    cosine_distance: float
    l2_distance: float
    interpretation: str


def model_dir_complete(model_path: Path) -> bool:
    if not model_path.exists() or not model_path.is_dir():
        return False
    return (model_path / "config.json").exists() and any(model_path.glob("*.safetensors"))


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _cell_name(row_idx: int, col_idx: int) -> str:
    return f"{chr(ord('A') + row_idx)}{col_idx + 1}"


def _interpret_feature_shift(cosine_distance: float, l2_distance: float) -> str:
    if cosine_distance >= 0.22:
        return "strong semantic feature shift; likely meaningful surface/layout change"
    if cosine_distance >= 0.14:
        return "moderate semantic feature shift; possible construction, grading, road change, or layout rework"
    if cosine_distance >= 0.08:
        return "weak to moderate feature shift; likely local surface disturbance or texture change"
    return "low semantic feature shift; mostly stable visual semantics"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def _colorize_region_map(region_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    rgb = np.zeros((*region_map.shape, 3), dtype=np.uint8)
    for idx, label in id2label.items():
        rgb[region_map == idx] = _hex_to_rgb(DINO_REGION_COLORS[label])
    return rgb


def _summarize_region_labels(region_map: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    total = max(int(region_map.size), 1)
    rows = []
    for idx, label in id2label.items():
        pixels = int((region_map == idx).sum())
        rows.append(
            {
                "id": int(idx),
                "label": label,
                "pixels": pixels,
                "percent": round(pixels * 100.0 / total, 3),
                "color": DINO_REGION_COLORS[label],
            }
        )
    return rows


def _kmeans(features: torch.Tensor, clusters: int, iterations: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    clusters = max(1, min(int(clusters), int(features.shape[0])))
    init_idx = torch.linspace(0, features.shape[0] - 1, clusters, device=features.device).long()
    centroids = features[init_idx].clone()
    labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(iterations):
        distances = torch.cdist(features, centroids)
        labels = distances.argmin(dim=1)
        next_centroids = []
        for cluster_idx in range(clusters):
            mask = labels == cluster_idx
            next_centroids.append(features[mask].mean(dim=0) if mask.any() else centroids[cluster_idx])
        centroids = torch.nn.functional.normalize(torch.stack(next_centroids, dim=0), dim=-1)
    return labels, centroids


def _resize_label_map(label_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(label_map.astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.NEAREST), dtype=np.int32)


def _artifact_from_region_map(region_map: np.ndarray, model_name: str, shared_cluster_space: bool) -> SegmentationArtifact:
    id2label = {idx: label for idx, label in DINO_REGION_ID2LABEL.items() if idx <= int(region_map.max(initial=0))}
    color_map = _colorize_region_map(region_map, id2label)
    return SegmentationArtifact(
        semantic_map=region_map.astype(np.int32),
        confidence_map=None,
        label_summary=_summarize_region_labels(region_map, id2label),
        overlay_rgb=color_map,
        legend=id2label,
        class_colors={label: DINO_REGION_COLORS[label] for label in id2label.values()},
        metadata={
            "backend": "dinov3_feature_regions",
            "model": model_name,
            "confidence_available": False,
            "semantic_classes": False,
            "shared_cluster_space": shared_cluster_space,
            "note": "DINOv3 has no land-cover segmentation head here; map shows unsupervised feature regions.",
        },
    )


class DINOv3FeatureEncoder:
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type in {"mps", "cuda"} else torch.float32
        repo_id = model_name_or_path
        if repo_id.startswith("hf_hub:"):
            repo_id = repo_id[len("hf_hub:") :]
        self.repo_id = repo_id
        self.model = timm.create_model(f"hf_hub:{self.repo_id}", pretrained=True, num_classes=0)
        if self.dtype == torch.float16:
            self.model = self.model.to(dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.data_config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.data_config, is_training=False)

    def unload(self) -> None:
        self.model = None
        self.transform = None
        if self.device.type == "mps" and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(_to_uint8_rgb(image))
        pixel_values = self.transform(pil).unsqueeze(0).to(self.device)
        if self.dtype == torch.float16:
            pixel_values = pixel_values.to(dtype=self.dtype)
        embedding = self.model.forward_features(pixel_values)
        if isinstance(embedding, (list, tuple)):
            embedding = embedding[0]
        if getattr(embedding, "ndim", 0) == 3:
            embedding = embedding[:, 0, :]
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].detach().cpu()

    @torch.no_grad()
    def patch_features(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        pil = Image.fromarray(_to_uint8_rgb(image))
        pixel_values = self.transform(pil).unsqueeze(0).to(self.device)
        if self.dtype == torch.float16:
            pixel_values = pixel_values.to(dtype=self.dtype)
        tokens = self.model.forward_features(pixel_values)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if getattr(tokens, "ndim", 0) != 3:
            raise ValueError(f"expected patch tokens as [B,N,C], got shape={getattr(tokens, 'shape', None)}")
        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1) or 0)
        patch_tokens = tokens[:, prefix_tokens:, :]
        patch_count = int(patch_tokens.shape[1])
        grid = int(round(patch_count**0.5))
        if grid * grid != patch_count:
            raise ValueError(f"cannot infer square patch grid from {patch_count} tokens")
        features = torch.nn.functional.normalize(patch_tokens[0].detach().float().cpu(), dim=-1)
        return features, (grid, grid)

    @torch.no_grad()
    def analyze_grid(self, before_rgb: np.ndarray, after_rgb: np.ndarray, grid_size: int = 4) -> list[DINOCellShift]:
        before_arr = _to_uint8_rgb(before_rgb)
        after_arr = _to_uint8_rgb(after_rgb)
        height, width = before_arr.shape[:2]
        row_bounds = np.linspace(0, height, grid_size + 1, dtype=int)
        col_bounds = np.linspace(0, width, grid_size + 1, dtype=int)
        rows: list[DINOCellShift] = []
        for row_idx in range(grid_size):
            for col_idx in range(grid_size):
                top = int(row_bounds[row_idx])
                bottom = int(row_bounds[row_idx + 1])
                left = int(col_bounds[col_idx])
                right = int(col_bounds[col_idx + 1])
                before_tile = before_arr[top:bottom, left:right]
                after_tile = after_arr[top:bottom, left:right]
                emb_before = self.encode(before_tile)
                emb_after = self.encode(after_tile)
                cosine = 1.0 - float(torch.nn.functional.cosine_similarity(emb_before.unsqueeze(0), emb_after.unsqueeze(0)).item())
                l2 = float(torch.norm(emb_before - emb_after, p=2).item())
                rows.append(
                    DINOCellShift(
                        cell=_cell_name(row_idx, col_idx),
                        cosine_distance=cosine,
                        l2_distance=l2,
                        interpretation=_interpret_feature_shift(cosine, l2),
                    )
                )
        rows.sort(key=lambda item: item.cosine_distance, reverse=True)
        return rows

    @torch.no_grad()
    def segment_pair_regions(
        self,
        before_rgb: np.ndarray,
        after_rgb: np.ndarray,
        clusters: int = 6,
    ) -> tuple[SegmentationArtifact, SegmentationArtifact]:
        before_arr = _to_uint8_rgb(before_rgb)
        after_arr = _to_uint8_rgb(after_rgb)
        before_features, before_grid = self.patch_features(before_arr)
        after_features, after_grid = self.patch_features(after_arr)
        if before_grid != after_grid:
            raise ValueError(f"DINO patch grids must match, got {before_grid} vs {after_grid}")

        combined = torch.cat([before_features, after_features], dim=0)
        labels, _ = _kmeans(combined, clusters=clusters)
        patch_count = before_features.shape[0]
        before_labels = labels[:patch_count].reshape(before_grid).numpy().astype(np.int32)
        after_labels = labels[patch_count:].reshape(after_grid).numpy().astype(np.int32)
        before_map = _resize_label_map(before_labels, (before_arr.shape[1], before_arr.shape[0]))
        after_map = _resize_label_map(after_labels, (after_arr.shape[1], after_arr.shape[0]))
        return (
            _artifact_from_region_map(before_map, self.repo_id, shared_cluster_space=True),
            _artifact_from_region_map(after_map, self.repo_id, shared_cluster_space=True),
        )
