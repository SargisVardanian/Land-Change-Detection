from __future__ import annotations

import argparse
from pathlib import Path

from land_change_detection.change_retrieval_datasets import IndexedChangeDataset, load_change_samples


def _mask_to_rgb(mask, size: tuple[int, int]):
    import numpy as np
    from PIL import Image

    if mask is None:
        return Image.new("RGB", size, color=(30, 30, 30))
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8) * 255
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = arr
    rgb[..., 1] = arr // 3
    return Image.fromarray(rgb).resize(size)


def _labeled(image: Image.Image, title: str) -> Image.Image:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (image.width, image.height + 28), color=(12, 12, 12))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a T1/T2/mask/caption preview from an indexed change dataset.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from PIL import Image, ImageDraw

    dataset = IndexedChangeDataset(load_change_samples(args.index))
    sample = next((row for row in dataset if row["sample_id"] == args.sample_id), None)
    if sample is None:
        raise ValueError(f"Sample not found: {args.sample_id}")

    before = Image.fromarray(sample["before"]).convert("RGB")
    after = Image.fromarray(sample["after"]).convert("RGB")
    mask = _mask_to_rgb(sample["mask"], before.size)

    before = _labeled(before, "T1 / before")
    after = _labeled(after, "T2 / after")
    mask = _labeled(mask, "change mask")

    width = before.width + after.width + mask.width
    height = max(before.height, after.height, mask.height) + 56
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width, 0))
    canvas.paste(mask, (before.width + after.width, 0))
    draw = ImageDraw.Draw(canvas)
    caption = sample["caption"] or "(no caption found)"
    draw.text((8, max(before.height, after.height, mask.height) + 10), f"{sample['sample_id']}: {caption}", fill=(255, 255, 255))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Rendered preview: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
