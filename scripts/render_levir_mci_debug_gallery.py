from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from land_change_detection.levir_mci import discover_levir_mci_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LEVIR-MCI debug gallery with abs-diff and changed-area ratio.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _thumb_rgb(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _thumb_mask(path: str, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    rgb = np.stack([arr, arr // 2, arr // 4], axis=-1)
    image = Image.fromarray(rgb)
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _thumb_abs_diff(before_path: str, after_path: str, size: tuple[int, int]) -> Image.Image:
    before = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.int16)
    after = np.asarray(Image.open(after_path).convert("RGB"), dtype=np.int16)
    diff = np.abs(after - before).astype(np.uint8)
    image = Image.fromarray(diff)
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> int:
    args = parse_args()
    samples = discover_levir_mci_samples(args.data_root)[: args.limit]
    tile_size = (144, 144)
    row_height = 220
    width = tile_size[0] * 4
    canvas = Image.new("RGB", (width, row_height * max(len(samples), 1)), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    for row_index, sample in enumerate(samples):
        y = row_index * row_height
        before = _thumb_rgb(sample.image_before, tile_size)
        after = _thumb_rgb(sample.image_after, tile_size)
        mask = _thumb_mask(sample.binary_change_mask, tile_size)
        diff = _thumb_abs_diff(sample.image_before, sample.image_after, tile_size)
        for col_index, image in enumerate((before, after, mask, diff)):
            x = col_index * tile_size[0]
            canvas.paste(image, (x, y))
        mask_arr = np.asarray(Image.open(sample.binary_change_mask).convert("L"))
        changed_area_ratio = float((mask_arr > 0).mean())
        draw.text((8, y + tile_size[1] + 8), f"{sample.sample_id} | {sample.split} | changed_area_ratio={changed_area_ratio:.4f}", fill=(255, 255, 255))
        draw.text((8, y + tile_size[1] + 26), sample.caption[:110], fill=(200, 200, 200))
    output = args.output or (args.project_root / "runs" / "levir_mci_debug_gallery.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Rendered debug gallery: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
