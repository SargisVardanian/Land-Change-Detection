#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _plot(rows: list[dict], keys: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in keys:
        ys = [row.get(key) for row in rows if row.get(key) is not None]
        if ys:
            ax.plot(range(len(ys)), ys, label=key)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    batch_rows = _read_jsonl(args.run_dir / "metrics_history.jsonl")
    epoch_rows = _read_jsonl(args.run_dir / "epoch_metrics.jsonl")
    plot_dir = args.run_dir / "plots"
    _plot(batch_rows, ["loss", "retrieval_loss", "semantic_loss", "local_loss", "mask_bce_loss", "mask_dice_loss", "text_mask_loss", "direction_loss"], plot_dir / "losses.png", "Losses")
    _plot(epoch_rows, ["R@1", "R@5", "R@10"], plot_dir / "retrieval.png", "Dataset Retrieval")
    _plot(epoch_rows, ["union_mask_dice", "text_mask_dice"], plot_dir / "masks.png", "Mask Metrics")
    _plot(batch_rows, ["lr", "grad_norm"], plot_dir / "optimization.png", "Optimization")


if __name__ == "__main__":
    main()
