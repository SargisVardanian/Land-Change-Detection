from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset


BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
RGB_INDICES = [3, 2, 1]
SATLAS_MS9_INDICES = [3, 2, 1, 4, 5, 6, 7, 11, 12]


@dataclass
class Scene:
    city: str
    pre: np.ndarray
    post: np.ndarray
    mask: np.ndarray
    band_mode: str = "rgb"

    @property
    def pre_rgb(self) -> np.ndarray:
        return np.moveaxis(select_bands(self.pre, "rgb"), 0, -1)

    @property
    def post_rgb(self) -> np.ndarray:
        return np.moveaxis(select_bands(self.post, "rgb"), 0, -1)


def robust_scale(arr: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    scaled = np.empty_like(arr, dtype=np.float32)
    for channel in range(arr.shape[0]):
        values = arr[channel]
        lo, hi = np.percentile(values, [lower, upper])
        if hi <= lo:
            hi = lo + 1.0
        scaled[channel] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return scaled


def select_bands(arr: np.ndarray, band_mode: str) -> np.ndarray:
    if arr.shape[0] == 3 and band_mode == "rgb":
        return arr
    if arr.shape[0] < 13:
        raise ValueError(f"expected 13-band input for '{band_mode}', got shape={arr.shape}")
    if band_mode == "rgb":
        return arr[RGB_INDICES]
    if band_mode == "ms9":
        return arr[SATLAS_MS9_INDICES]
    if band_mode == "all13":
        return arr
    raise ValueError(f"unsupported band_mode={band_mode}")


def find_single_root(root: Path, prefix: str) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one directory starting with '{prefix}' under {root}")
    return matches[0]


def read_stack(folder: Path) -> np.ndarray:
    tif_candidates = list(folder.glob("*.tif"))
    band_tifs = {path.stem: path for path in tif_candidates if path.stem in BAND_NAMES}
    if len(band_tifs) == len(BAND_NAMES):
        arr = np.stack([tifffile.imread(band_tifs[band]) for band in BAND_NAMES], axis=0)
    else:
        npy_candidates = sorted(folder.glob("*.npy"))
        if npy_candidates:
            arr = np.load(npy_candidates[0])
        elif tif_candidates:
            arr = tifffile.imread(sorted(tif_candidates)[0])
        else:
            png_candidates = sorted(folder.glob("*.png"))
            band_pngs = {path.stem: path for path in png_candidates if path.stem in BAND_NAMES}
            if len(band_pngs) == len(BAND_NAMES):
                arr = np.stack([np.array(Image.open(band_pngs[band])) for band in BAND_NAMES], axis=0)
            elif len(png_candidates) == 1:
                arr = np.array(Image.open(png_candidates[0]))
            elif png_candidates:
                arr = np.stack([np.array(Image.open(path)) for path in png_candidates], axis=0)
            else:
                raise FileNotFoundError(f"no raster data found in {folder}")

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] not in (3, 9, 13):
        arr = np.moveaxis(arr, -1, 0)
    return arr.astype(np.float32)


def read_mask(mask_dir: Path) -> np.ndarray:
    tif_files = sorted(mask_dir.glob("*-cm.tif"))
    if tif_files:
        arr = tifffile.imread(tif_files[0]).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.min() >= 1:
            arr = arr - arr.min()
        return (arr > 0).astype(np.float32)
    png = np.array(Image.open(mask_dir / "cm.png"), dtype=np.float32)
    return (png > 0).astype(np.float32)


class OSCDSceneRepository:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images_root = find_single_root(self.root / "images", "Onera Satellite Change Detection dataset - Images")
        self.train_root = find_single_root(
            self.root / "train_labels",
            "Onera Satellite Change Detection dataset - Train Labels",
        )
        self.test_root = find_single_root(
            self.root / "test_labels",
            "Onera Satellite Change Detection dataset - Test Labels",
        )
        self.train_cities = sorted(path.name for path in self.train_root.iterdir() if path.is_dir())
        self.test_cities = sorted(path.name for path in self.test_root.iterdir() if path.is_dir())
        self._cache: dict[str, Scene] = {}

    def split_cities(self, split: str) -> list[str]:
        if split == "train":
            return self.train_cities
        if split == "test":
            return self.test_cities
        if split == "all":
            return sorted(self.train_cities + self.test_cities)
        raise ValueError(f"unsupported split={split}")

    def label_root_for_city(self, city: str) -> Path:
        if city in self.train_cities:
            return self.train_root / city
        return self.test_root / city

    def load_scene(self, city: str, *, normalize: bool = True) -> Scene:
        if city in self._cache:
            return self._cache[city]

        image_city_root = self.images_root / city
        label_city_root = self.label_root_for_city(city)
        pre = read_stack(image_city_root / "imgs_1_rect")
        post = read_stack(image_city_root / "imgs_2_rect")
        mask = read_mask(label_city_root / "cm")
        if normalize:
            pre = robust_scale(pre)
            post = robust_scale(post)
        scene = Scene(city=city, pre=pre, post=post, mask=mask)
        self._cache[city] = scene
        return scene


def sliding_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def pad_spatial(arr: np.ndarray, tile_size: int, fill_value: float = 0.0) -> np.ndarray:
    if arr.ndim == 3:
        channels, height, width = arr.shape
        if height == tile_size and width == tile_size:
            return arr
        padded = np.full((channels, tile_size, tile_size), fill_value, dtype=arr.dtype)
        padded[:, :height, :width] = arr
        return padded
    if arr.ndim == 2:
        height, width = arr.shape
        if height == tile_size and width == tile_size:
            return arr
        padded = np.full((tile_size, tile_size), fill_value, dtype=arr.dtype)
        padded[:height, :width] = arr
        return padded
    raise ValueError(f"unsupported array shape for padding: {arr.shape}")


class OSCDPatchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        band_mode: str,
        tile_size: int = 256,
        stride: int = 192,
        train: bool = True,
        samples_per_epoch: int = 768,
    ):
        self.repo = OSCDSceneRepository(root)
        self.cities = self.repo.split_cities(split)
        self.band_mode = band_mode
        self.tile_size = tile_size
        self.stride = stride
        self.train = train
        self.samples_per_epoch = samples_per_epoch
        self.eval_tiles: list[tuple[str, int, int]] = []
        if not self.train:
            for city in self.cities:
                scene = self.repo.load_scene(city)
                height, width = scene.mask.shape
                for top in sliding_positions(height, tile_size, stride):
                    for left in sliding_positions(width, tile_size, stride):
                        self.eval_tiles.append((city, top, left))

    def __len__(self) -> int:
        return self.samples_per_epoch if self.train else len(self.eval_tiles)

    def _random_tile(self, city: str) -> tuple[int, int]:
        scene = self.repo.load_scene(city)
        height, width = scene.mask.shape
        tile = self.tile_size
        positive = np.argwhere(scene.mask > 0.5)
        if len(positive) > 0 and np.random.rand() < 0.6:
            center_y, center_x = positive[np.random.randint(0, len(positive))]
            top = int(np.clip(center_y - tile // 2, 0, max(0, height - tile)))
            left = int(np.clip(center_x - tile // 2, 0, max(0, width - tile)))
        else:
            top = np.random.randint(0, max(1, height - tile + 1))
            left = np.random.randint(0, max(1, width - tile + 1))
        return top, left

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.train:
            city = self.cities[np.random.randint(0, len(self.cities))]
            top, left = self._random_tile(city)
        else:
            city, top, left = self.eval_tiles[index]

        scene = self.repo.load_scene(city)
        pre = select_bands(scene.pre, self.band_mode)
        post = select_bands(scene.post, self.band_mode)
        bottom = top + self.tile_size
        right = left + self.tile_size

        pre_patch = pre[:, top:bottom, left:right]
        post_patch = post[:, top:bottom, left:right]
        mask_patch = scene.mask[top:bottom, left:right]

        pre_patch = pad_spatial(pre_patch, self.tile_size)
        post_patch = pad_spatial(post_patch, self.tile_size)
        mask_patch = pad_spatial(mask_patch, self.tile_size)[None, :, :]

        return {
            "pre": torch.from_numpy(pre_patch.astype(np.float32)),
            "post": torch.from_numpy(post_patch.astype(np.float32)),
            "mask": torch.from_numpy(mask_patch.astype(np.float32)),
            "city": city,
        }
