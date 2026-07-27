#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def _show_image(ax: plt.Axes, tensor: Tensor, title: str) -> None:
    image = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def _show_mask(ax: plt.Axes, mask: Tensor, title: str) -> None:
    ax.imshow(mask.detach().cpu().float().numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def render_training_panel(
    output_path: str | Path,
    t1: Tensor,
    t2: Tensor,
    gt_mask: Tensor,
    predicted_union_mask: Tensor,
    event_masks: Tensor,
    text_conditioned_mask: Tensor,
    caption: str,
    retrieval_lines: list[str] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    _show_image(axes[0, 0], t1, "T1")
    _show_image(axes[0, 1], t2, "T2")
    _show_mask(axes[0, 2], gt_mask, "GT change mask")
    _show_mask(axes[0, 3], predicted_union_mask, "Predicted union")
    active = event_masks[: min(3, event_masks.shape[0])]
    for idx in range(3):
        if idx < active.shape[0]:
            _show_mask(axes[1, idx], active[idx], f"Event {idx}")
        else:
            axes[1, idx].axis("off")
    _show_mask(axes[1, 3], text_conditioned_mask, "Text-conditioned")
    fig.suptitle(caption[:160])
    if retrieval_lines:
        fig.text(0.02, 0.01, "\n".join(retrieval_lines[:5]), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an index page for UniChange joint visual artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visuals = args.run_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    panels = sorted(visuals.glob("**/*.png"))
    rows = "\n".join(f'<figure><img src="{panel.relative_to(visuals)}"><figcaption>{panel.name}</figcaption></figure>' for panel in panels)
    (visuals / "index.html").write_text(f"<html><body><h1>UniChange Joint Visuals</h1>{rows}</body></html>")


if __name__ == "__main__":
    main()
