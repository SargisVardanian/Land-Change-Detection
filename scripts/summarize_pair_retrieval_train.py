from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pair-retrieval training history into a compact JSON artifact.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _score(row: dict[str, Any]) -> float:
    eval_metrics = dict(row.get("eval", {}))
    return float(
        eval_metrics.get("recall@5", 0.0)
        + eval_metrics.get("mAP", 0.0)
        + eval_metrics.get("transition_recall@5", 0.0)
        + eval_metrics.get("transition_top1_hit_rate", 0.0)
    )


def _highlights(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics.get("loss"),
        "recall@1": metrics.get("recall@1"),
        "recall@5": metrics.get("recall@5"),
        "recall@10": metrics.get("recall@10"),
        "mAP": metrics.get("mAP"),
        "MRR": metrics.get("MRR"),
        "transition_recall@1": metrics.get("transition_recall@1"),
        "transition_recall@5": metrics.get("transition_recall@5"),
        "transition_recall@10": metrics.get("transition_recall@10"),
        "transition_MRR": metrics.get("transition_MRR"),
        "transition_top1_hit_rate": metrics.get("transition_top1_hit_rate"),
        "mean_transition_similarity_top5": metrics.get("mean_transition_similarity_top5"),
        "pair_sample_fraction": metrics.get("pair_sample_fraction"),
    }


def build_summary(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics_history.json").read_text(encoding="utf-8"))
    history = list(payload.get("history", []))
    best_row = max(history, key=_score, default={})
    latest_row = history[-1] if history else {}
    best_epoch = best_row.get("epoch")
    summary = {
        "run_dir": str(run_dir),
        "num_epochs": len(history),
        "best_epoch": best_epoch,
        "best_score": _score(best_row) if best_row else None,
        "best_eval_highlights": _highlights(dict(best_row.get("eval", {}))) if best_row else {},
        "latest_train_highlights": _highlights(dict(latest_row.get("train", {}))) if latest_row else {},
        "latest_eval_highlights": _highlights(dict(latest_row.get("eval", {}))) if latest_row else {},
        "best_checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
        "last_checkpoint": str(run_dir / "last.pt") if (run_dir / "last.pt").exists() else None,
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = build_summary(args.run_dir)
    rendered = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
