from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UniChange v2 training and retrieval metrics.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_loss(step_rows: list[dict], output: Path) -> None:
    if not step_rows:
        return
    steps = [int(row["step"]) for row in step_rows]
    losses = [float(row["loss"]) for row in step_rows]
    window = min(50, max(1, len(losses) // 20))
    rolling = []
    for index in range(len(losses)):
        start = max(0, index - window + 1)
        rolling.append(sum(losses[start:index + 1]) / (index - start + 1))

    figure = plt.figure(figsize=(10, 5.5))
    plt.plot(steps, losses, alpha=0.25, label="Step loss")
    plt.plot(steps, rolling, linewidth=2, label=f"Rolling mean ({window})")
    plt.xlabel("Training step")
    plt.ylabel("Contrastive loss")
    plt.title("UniChange v2 training loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_retrieval(epoch_rows: list[dict], output: Path) -> None:
    if not epoch_rows:
        return
    epochs = [int(row["epoch"]) for row in epoch_rows]
    figure = plt.figure(figsize=(10, 5.5))
    for key, label in (
        ("text_to_pair_R@1", "R@1"),
        ("text_to_pair_R@5", "R@5"),
        ("text_to_pair_R@10", "R@10"),
        ("MRR", "MRR"),
    ):
        if all(key in row for row in epoch_rows):
            plt.plot(epochs, [float(row[key]) for row in epoch_rows], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("Duplicate-aware text-to-pair retrieval")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_efficiency(epoch_rows: list[dict], output: Path) -> None:
    if not epoch_rows:
        return
    epochs = [int(row["epoch"]) for row in epoch_rows]
    train_throughput = [float(row.get("train_pairs_per_second", 0.0)) for row in epoch_rows]
    retrieval_throughput = [float(row.get("pairs_per_second", 0.0)) for row in epoch_rows]
    search_throughput = [float(row.get("search_queries_per_second", 0.0)) for row in epoch_rows]

    figure = plt.figure(figsize=(10, 5.5))
    plt.plot(epochs, train_throughput, marker="o", label="Training pairs/s")
    plt.plot(epochs, retrieval_throughput, marker="o", label="Validation pairs/s")
    plt.plot(epochs, search_throughput, marker="o", label="Search queries/s")
    plt.xlabel("Epoch")
    plt.ylabel("Throughput")
    plt.title("Training and retrieval efficiency")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.history)
    step_rows = [row for row in rows if row.get("record_type") in (None, "train_step") and "loss" in row]
    epoch_rows = [row for row in rows if row.get("record_type") == "epoch"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(step_rows, args.output_dir / "loss_curve.png")
    plot_retrieval(epoch_rows, args.output_dir / "retrieval_metrics.png")
    plot_efficiency(epoch_rows, args.output_dir / "efficiency.png")

    best = max(epoch_rows, key=lambda row: float(row.get("text_to_pair_R@1", -1.0))) if epoch_rows else None
    summary = {
        "history": str(args.history),
        "step_records": len(step_rows),
        "epoch_records": len(epoch_rows),
        "best_epoch": int(best["epoch"]) if best else None,
        "best_metrics": best,
        "final_epoch": epoch_rows[-1] if epoch_rows else None,
        "plots": {
            "loss": str(args.output_dir / "loss_curve.png"),
            "retrieval": str(args.output_dir / "retrieval_metrics.png"),
            "efficiency": str(args.output_dir / "efficiency.png"),
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
