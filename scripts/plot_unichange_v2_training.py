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
    parser.add_argument("--resolved-config", type=Path, default=None)
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


def _rolling(values: list[float], window: int = 25) -> list[float]:
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(values))]


def plot_training_objectives(step_rows: list[dict], output: Path) -> None:
    if not step_rows:
        return
    steps = [int(row["step"]) for row in step_rows]
    groups = (
        (("loss", "Total"), ("retrieval_loss", "Retrieval")),
        (("local_positive_negative_margin_loss", "Local margin"), ("conditional_instance_discrimination_loss", "Conditional identity")),
        (("query_segmentation_loss", "Query mask"), ("changed_channel_loss", "Changed"), ("appeared_channel_loss", "Appeared"), ("disappeared_channel_loss", "Disappeared")),
        (("structured_auxiliary_evidence_loss", "Structured attributes"), ("temporal_reversal_consistency_loss", "Temporal reversal")),
    )
    titles = ("Ranking objectives", "Fine-grained discrimination", "Localization objectives", "Structured and temporal objectives")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, title, series in zip(axes.flat, titles, groups, strict=True):
        for key, label in series:
            values = [float(row.get(key, 0.0)) for row in step_rows]
            axis.plot(steps, _rolling(values), label=label)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_optimization_diagnostics(step_rows: list[dict], output: Path, grad_clip_norm: float | None) -> None:
    if not step_rows:
        return
    steps = [int(row["step"]) for row in step_rows]
    norms = [float(row.get("grad_norm_before_clip", 0.0)) for row in step_rows]
    clipped = [float(bool(row.get("gradient_was_clipped", False))) for row in step_rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(steps, norms, alpha=0.25, label="Raw gradient norm")
    axes[0].plot(steps, _rolling(norms), linewidth=2, label="Rolling mean")
    if grad_clip_norm is not None:
        axes[0].axhline(grad_clip_norm, color="red", linestyle="--", label=f"Clip threshold {grad_clip_norm:g}")
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("Gradient norm before clipping")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, _rolling(clipped), color="darkorange", label="Rolling clipping fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Fraction")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_hard_negative_coverage(step_rows: list[dict], output: Path) -> None:
    categories = (
        ("same_object_wrong_direction", "Wrong direction"),
        ("same_object_direction_wrong_location", "Wrong location"),
        ("same_object_location_wrong_count", "Wrong count"),
        ("same_broad_change_wrong_object", "Wrong object"),
        ("no_change_lookalike", "No-change lookalike"),
    )
    totals = [sum(int(row.get(f"hard_negative_{key}_count", 0)) for row in step_rows) for key, _ in categories]
    figure = plt.figure(figsize=(11, 5.5))
    bars = plt.bar([label for _, label in categories], totals)
    plt.bar_label(bars, padding=3)
    plt.ylabel("Selected negatives across training")
    plt.title("Structured hard-negative coverage")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_module_gradient_norms(step_rows: list[dict], output: Path) -> None:
    if not step_rows:
        return
    steps = [int(row["step"]) for row in step_rows]
    keys = (
        ("grad_norm_temporal_encoder", "Temporal encoder"),
        ("grad_norm_text_adapter", "Text adapter"),
        ("grad_norm_qcpr_interaction", "QCPR interaction"),
        ("grad_norm_qcpr_temporal_channel", "Temporal channel head"),
        ("grad_norm_qcpr_fusion_calibration", "Fusion/calibration"),
    )
    available = [(key, label) for key, label in keys if any(key in row for row in step_rows)]
    if not available:
        return
    figure = plt.figure(figsize=(12, 6.5))
    for key, label in available:
        values = [float(row.get(key, 0.0)) for row in step_rows]
        plt.plot(steps, _rolling(values), label=label)
    plt.xlabel("Step")
    plt.ylabel("Gradient L2 norm before clipping")
    plt.title("Per-module gradient norms")
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
    resolved_config = json.loads(args.resolved_config.read_text()) if args.resolved_config and args.resolved_config.exists() else {}

    plot_loss(step_rows, args.output_dir / "loss_curve.png")
    plot_retrieval(epoch_rows, args.output_dir / "retrieval_metrics.png")
    plot_efficiency(epoch_rows, args.output_dir / "efficiency.png")
    plot_training_objectives(step_rows, args.output_dir / "training_objectives.png")
    plot_optimization_diagnostics(step_rows, args.output_dir / "optimization_diagnostics.png", resolved_config.get("grad_clip_norm"))
    plot_hard_negative_coverage(step_rows, args.output_dir / "hard_negative_coverage.png")
    plot_module_gradient_norms(step_rows, args.output_dir / "module_gradient_norms.png")

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
            "training_objectives": str(args.output_dir / "training_objectives.png"),
            "optimization_diagnostics": str(args.output_dir / "optimization_diagnostics.png"),
            "hard_negative_coverage": str(args.output_dir / "hard_negative_coverage.png"),
            "module_gradient_norms": str(args.output_dir / "module_gradient_norms.png"),
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
