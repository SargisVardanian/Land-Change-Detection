from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run metrics from a training output directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = args.run_dir / "metrics_history.json"
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = list(payload.get("history", []))
    best_row = max(history, key=lambda row: row.get("eval", {}).get("dice", row.get("eval", {}).get("recall@1", 0.0)), default={})
    latest = history[-1] if history else {}
    summary = {
        "run_dir": str(args.run_dir),
        "best_dice": best_row.get("eval", {}).get("dice"),
        "best_recall@1": best_row.get("eval", {}).get("recall@1"),
        "best_recall@5": best_row.get("eval", {}).get("recall@5"),
        "best_recall@10": best_row.get("eval", {}).get("recall@10"),
        "latest_epoch_metrics": latest,
        "best_checkpoint": str(args.run_dir / "best.pt") if (args.run_dir / "best.pt").exists() else None,
    }
    rendered = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
