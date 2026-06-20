from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LEVIR-CC retrieval overfit/sanity experiment.")
    parser.add_argument("--levir-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--preset", default="simple_patch_smoke")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-train-samples", type=int, default=100)
    parser.add_argument("--min-r5-improvement", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _run_train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/train_dino_pair_retrieval.py",
        "--preset",
        args.preset,
        "--levir-manifest",
        str(args.levir_manifest),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--max-train-samples",
        str(args.max_train_samples),
        "--device",
        args.device,
    ]
    if args.project_root is not None:
        cmd.extend(["--project-root", str(args.project_root)])
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, env={"PYTHONPATH": "src"})
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)


def _manifest_counts(path: Path) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows), len({str(row.get("pair_id") or row["sample_id"]) for row in rows})


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    caption_row_count, unique_pair_count = _manifest_counts(args.levir_manifest)
    _run_train(args)
    history_payload = json.loads((args.output_dir / "metrics_history.json").read_text(encoding="utf-8"))
    history = list(history_payload.get("history", []))
    if not history:
        raise SystemExit("Training produced no metrics history.")
    start_r5 = float(history[0].get("eval", {}).get("recall@5", 0.0))
    best_r5 = max(float(row.get("eval", {}).get("recall@5", 0.0)) for row in history)
    improved = best_r5 >= start_r5 + args.min_r5_improvement
    report = {
        "preset": args.preset,
        "start_recall@5": start_r5,
        "best_recall@5": best_r5,
        "improved_recall@5": improved,
        "epochs": len(history),
        "max_train_samples": args.max_train_samples,
        "caption_row_count": caption_row_count,
        "unique_pair_count": unique_pair_count,
        "best_checkpoint": str(args.output_dir / "best.pt") if (args.output_dir / "best.pt").exists() else None,
    }
    (args.output_dir / "overfit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not improved:
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
