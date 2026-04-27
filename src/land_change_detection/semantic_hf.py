from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MASK2FORMER_SATELLITE_DIR = Path("artifacts/models/semantic/mask2former-satellite")
PRITHVI_EO_300M_TL_DIR = Path("artifacts/models/semantic/Prithvi-EO-2.0-300M-TL")
PRITHVI_EO_600M_TL_DIR = Path("artifacts/models/semantic/Prithvi-EO-2.0-600M-TL")
PRITHVI_EO_DIR = PRITHVI_EO_300M_TL_DIR

DEFAULT_CLASS_COLORS: dict[str, str] = {
    "background": "#111111",
    "bareland": "#b4a08a",
    "grass": "#9adf67",
    "pavement": "#7f8c8d",
    "road": "#34495e",
    "tree": "#1e8449",
    "water": "#2e86de",
    "cropland": "#f4d03f",
    "building": "#e74c3c",
}

DEFAULT_ID2LABEL: dict[int, str] = {
    0: "background",
    1: "bareland",
    2: "grass",
    3: "pavement",
    4: "road",
    5: "tree",
    6: "water",
    7: "cropland",
    8: "building",
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC RGB, got shape={arr.shape}")
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class SemanticSegmentationResult:
    class_map: np.ndarray
    color_map: np.ndarray
    label_summary: list[dict]


class Mask2FormerSatelliteSegmenter:
    def __init__(self, model_dir: str | Path = MASK2FORMER_SATELLITE_DIR, device: str | torch.device = "cpu"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Mask2Former satellite model not found: {self.model_dir}")
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None) or {}
        self.id2label = {int(k): str(v) for k, v in id2label.items()} if isinstance(next(iter(id2label.keys()), 0), str) else {int(k): str(v) for k, v in id2label.items()}
        if not self.id2label or all(str(label).startswith("LABEL_") for label in self.id2label.values()):
            self.id2label = DEFAULT_ID2LABEL.copy()
        self.label2id = {label: idx for idx, label in self.id2label.items()}

    @torch.no_grad()
    def segment(self, rgb: np.ndarray) -> SemanticSegmentationResult:
        rgb_uint8 = _to_uint8_rgb(rgb)
        inputs = self.processor(images=rgb_uint8, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        height, width = rgb_uint8.shape[:2]
        class_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
        class_map = class_map.detach().cpu().numpy().astype(np.int32)
        return SemanticSegmentationResult(
            class_map=class_map,
            color_map=colorize_semantic_map(class_map, self.id2label),
            label_summary=summarize_semantic_labels(class_map, self.id2label),
        )


def colorize_semantic_map(class_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for idx, label in id2label.items():
        color = DEFAULT_CLASS_COLORS.get(label, "#95a5a6")
        rgb[class_map == idx] = _hex_to_rgb(color)
    return rgb


def summarize_semantic_labels(class_map: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    total = max(int(class_map.size), 1)
    result: list[dict] = []
    for idx in sorted(id2label.keys()):
        pixels = int((class_map == idx).sum())
        result.append(
            {
                "id": int(idx),
                "label": id2label[idx],
                "pixels": pixels,
                "percent": round(pixels * 100.0 / total, 3),
                "color": DEFAULT_CLASS_COLORS.get(id2label[idx], "#95a5a6"),
            }
        )
    return result


def build_transition_overlay(before_map: np.ndarray, after_map: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    if before_map.shape != after_map.shape:
        raise ValueError(f"surface maps must match, got {before_map.shape} vs {after_map.shape}")
    base = _to_uint8_rgb(after_rgb).astype(np.float32)
    changed = before_map != after_map
    overlay = base.copy()
    overlay[changed] = base[changed] * 0.3 + np.array([231, 76, 60], dtype=np.float32) * 0.7
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return {
        "left": int(xs.min()),
        "top": int(ys.min()),
        "right": int(xs.max()),
        "bottom": int(ys.max()),
    }


def expand_bbox_to_square(
    bbox: dict[str, int],
    height: int,
    width: int,
    margin: int = 32,
) -> dict[str, int]:
    left = max(0, bbox["left"] - margin)
    top = max(0, bbox["top"] - margin)
    right = min(width - 1, bbox["right"] + margin)
    bottom = min(height - 1, bbox["bottom"] + margin)

    box_w = right - left + 1
    box_h = bottom - top + 1
    side = max(box_w, box_h)

    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half = side // 2
    new_left = max(0, cx - half)
    new_top = max(0, cy - half)
    new_right = min(width - 1, new_left + side - 1)
    new_bottom = min(height - 1, new_top + side - 1)

    if new_right - new_left + 1 < side:
        new_left = max(0, new_right - side + 1)
    if new_bottom - new_top + 1 < side:
        new_top = max(0, new_bottom - side + 1)

    return {
        "left": int(new_left),
        "top": int(new_top),
        "right": int(new_right),
        "bottom": int(new_bottom),
    }


def crop_rgb(rgb: np.ndarray, bbox: dict[str, int]) -> np.ndarray:
    arr = _to_uint8_rgb(rgb)
    return arr[bbox["top"] : bbox["bottom"] + 1, bbox["left"] : bbox["right"] + 1]


def draw_bbox(rgb: np.ndarray, bbox: dict[str, int], color: tuple[int, int, int] = (255, 80, 80), width: int = 4) -> np.ndarray:
    img = Image.fromarray(_to_uint8_rgb(rgb))
    draw = ImageDraw.Draw(img)
    for offset in range(width):
        draw.rectangle(
            [
                bbox["left"] - offset,
                bbox["top"] - offset,
                bbox["right"] + offset,
                bbox["bottom"] + offset,
            ],
            outline=color,
        )
    return np.asarray(img)


def build_focus_square(
    before_map: np.ndarray,
    after_map: np.ndarray,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    square_size: int = 512,
    stride: int = 128,
) -> dict[str, object]:
    changed = before_map != after_map
    height, width = before_map.shape
    square_size = int(min(square_size, height, width))
    if square_size <= 0:
        square_size = min(height, width)

    best_score = -1.0
    best_count = -1
    best_top = max(0, (height - square_size) // 2)
    best_left = max(0, (width - square_size) // 2)

    if changed.any():
        for top in range(0, max(1, height - square_size + 1), max(1, stride)):
            for left in range(0, max(1, width - square_size + 1), max(1, stride)):
                bottom = top + square_size
                right = left + square_size
                patch = changed[top:bottom, left:right]
                score = float(patch.mean())
                count = int(patch.sum())
                if score > best_score or (score == best_score and count > best_count):
                    best_score = score
                    best_count = count
                    best_top = top
                    best_left = left

    if height <= square_size:
        best_top = 0
    else:
        best_top = min(best_top, height - square_size)
    if width <= square_size:
        best_left = 0
    else:
        best_left = min(best_left, width - square_size)

    square = {
        "left": int(best_left),
        "top": int(best_top),
        "right": int(min(width - 1, best_left + square_size - 1)),
        "bottom": int(min(height - 1, best_top + square_size - 1)),
    }
    return {
        "bbox": square,
        "before_crop": crop_rgb(before_rgb, square),
        "after_crop": crop_rgb(after_rgb, square),
        "before_boxed": draw_bbox(before_rgb, square),
        "after_boxed": draw_bbox(after_rgb, square),
    }
