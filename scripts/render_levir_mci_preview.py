from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from land_change_detection.visualization import mask_rgba_overlay, rgb_absolute_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LEVIR-MCI preview panels from an evaluation manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def render_item(item: dict, output: Path, dpi: int) -> None:
    t1 = load_rgb(item["t1"])
    t2 = load_rgb(item["t2"])
    mask = load_mask(item["mask"])
    difference = rgb_absolute_difference(t1, t2)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(t1)
    axes[0].set_title("T1 / before")
    axes[1].imshow(t2)
    axes[1].set_title("T2 / after")
    axes[2].imshow(difference, cmap="magma")
    axes[2].set_title("RGB absolute difference")
    axes[3].imshow(t2)
    axes[3].imshow(mask_rgba_overlay(mask))
    axes[3].set_title("GT change mask")

    for axis in axes:
        axis.axis("off")

    title = (
        f"{item['pair_id']} | {item['stratum']} | "
        f"mask={float(item['mask_fraction']):.4f} | {item['query_caption']}"
    )
    figure.suptitle(title[:240])
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise RuntimeError(f"Manifest contains no items: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items):
        filename = f"{index:03d}_{item['stratum']}_{item['pair_id']}.png"
        render_item(item, args.output_dir / filename, args.dpi)

    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} previews to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
