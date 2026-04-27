from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


DINO_V3_VITS16 = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_V3_VITS16_DIR = Path("artifacts/models/features/timm-vit_small_patch16_dinov3.lvd1689m")
DINO_V3_VITB16 = "timm/vit_base_patch16_dinov3.lvd1689m"
DINO_V3_VITB16_DIR = Path("artifacts/models/features/timm-vit_base_patch16_dinov3.lvd1689m")
DINO_V3_VITL16_SAT = "timm/vit_large_patch16_dinov3.sat493m"
DINO_V3_VITL16_SAT_DIR = Path("artifacts/models/features/timm-vit_large_patch16_dinov3.sat493m")


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
