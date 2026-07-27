from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from land_change_detection.levir_mci import discover_levir_mci_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a grid of real LEVIR-MCI before/after/mask samples.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=9)
    return parser.parse_args()


def _thumb(path: str, size: tuple[int, int], binary: bool = False) -> Image.Image:
    image = Image.open(path)
    if binary:
        mask = image.convert("L")
        arr = (mask.point(lambda value: 255 if value > 0 else 0)).convert("RGB")
        image = arr
    else:
        image = image.convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 18, 18))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> int:
    args = parse_args()
    samples = discover_levir_mci_samples(args.data_root)
    if not samples:
        raise SystemExit("No LEVIR-MCI samples found.")
    selected = samples[: max(1, args.limit)]
    cell_size = (160, 160)
    caption_height = 48
    row_height = cell_size[1] + caption_height
    cols = 3
    canvas = Image.new("RGB", (cols * cell_size[0], len(selected) * row_height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)

    for row_index, sample in enumerate(selected):
        y = row_index * row_height
        tiles = [
            _thumb(sample.image_before, cell_size),
            _thumb(sample.image_after, cell_size),
            _thumb(sample.binary_change_mask, cell_size, binary=True),
        ]
        for col_index, tile in enumerate(tiles):
            canvas.paste(tile, (col_index * cell_size[0], y))
        label = f"{sample.split}:{sample.sample_id} | {sample.caption[:72]}"
        draw.text((8, y + cell_size[1] + 10), label, fill=(255, 255, 255))

    output = args.output or (args.project_root / "runs" / "levir_mci_grid.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Rendered grid: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
