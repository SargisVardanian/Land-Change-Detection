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
    parser.add_argument("--min-loss-drop", type=float, default=0.05)
    parser.add_argument("--min-recall-at-1", type=float, default=0.80)
    parser.add_argument("--min-recall-at-5", type=float, default=0.90)
    parser.add_argument("--min-recall-at-10", type=float, default=0.99)
    parser.add_argument("--min-pair-to-text-recall-at-5", type=float, default=0.50)
    parser.add_argument("--min-anchor-positive-ratio", type=float, default=0.95)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--remoteclip-checkpoint", type=Path, default=None)
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
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--device",
        args.device,
    ]
    if args.remoteclip_checkpoint is not None:
        cmd.extend(["--remoteclip-arch", args.remoteclip_arch, "--remoteclip-checkpoint", str(args.remoteclip_checkpoint)])
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
    eval_rows = [row.get("eval", {}) for row in history]
    start_loss = float(eval_rows[0].get("loss", 0.0))
    best_row = max(eval_rows, key=lambda row: float(row.get("recall@5", 0.0)) + float(row.get("recall@10", 0.0)))
    best_loss = float(best_row.get("loss", 0.0))
    best_r1 = float(best_row.get("recall@1", 0.0))
    best_r5 = float(best_row.get("recall@5", 0.0))
    best_r10 = float(best_row.get("recall@10", 0.0))
    best_pair_r5 = float(best_row.get("pair_to_text_recall@5", 0.0))
    best_anchor_ratio = float(best_row.get("anchors_with_positive_ratio", 0.0))
    finite_metrics = all(
        all(isinstance(value, (int, float)) and abs(float(value)) != float("inf") and float(value) == float(value) for value in row.values())
        for row in eval_rows
    )
    loss_decreased = best_loss <= start_loss * (1.0 - args.min_loss_drop)
    improved = (
        loss_decreased
        and best_r1 >= args.min_recall_at_1
        and best_r5 >= args.min_recall_at_5
        and best_r10 >= args.min_recall_at_10
        and best_pair_r5 >= args.min_pair_to_text_recall_at_5
        and best_anchor_ratio >= args.min_anchor_positive_ratio
        and finite_metrics
    )
    report = {
        "preset": args.preset,
        "start_loss": start_loss,
        "best_loss": best_loss,
        "best_recall@1": best_r1,
        "best_recall@5": best_r5,
        "best_recall@10": best_r10,
        "best_pair_to_text_recall@5": best_pair_r5,
        "best_anchors_with_positive_ratio": best_anchor_ratio,
        "loss_decreased_materially": loss_decreased,
        "finite_metrics": finite_metrics,
        "gate_passed": improved,
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
