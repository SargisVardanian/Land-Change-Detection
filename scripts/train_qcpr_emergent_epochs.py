#!/usr/bin/env python3
"""Epoch controller for mask-free A0 and B experiments.

Each epoch is an immutable invocation of the canonical step trainer followed by
development-only evaluation.  The controller never opens localization masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from land_change_detection.models.qcpr_v3_data import CappedCompositionalBatchSampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, allow_gate_failure: bool = False) -> int:
    print(json.dumps({"command": command}), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode and not allow_gate_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(report: dict[str, object], *path: str, default: float = math.nan) -> float:
    value: object = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_training_command(
    args: argparse.Namespace,
    *,
    epoch: int,
    total_steps: int,
    phase: str,
    resume_checkpoint: Path | None,
) -> list[str]:
    """Build one immutable epoch command without mixing warm start and resume."""
    command = [
        sys.executable, "scripts/train_qcpr_v3.py",
        "--output-dir", str(args.output_dir / "epochs" / f"epoch_{epoch:03d}"), "--phase", phase,
        "--initialization-mode", "clean_pretrained",
        "--derived-manifest-dir", str(args.manifest_dir),
        "--steps", str(total_steps), "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers), "--seed", str(args.seed),
        "--sampler-epoch", str(epoch - 1), "--checkpoint-interval", "250",
        "--contrastive-temperature", "0.07",
    ]
    if args.track == "B":
        if args.initialization_checkpoint is None:
            raise ValueError("B requires accepted A0 initialization checkpoint")
        command += ["--v3-checkpoint", str(args.initialization_checkpoint)]
        command += ["--b-stage", "B1" if epoch <= 10 else "B2"]
    if resume_checkpoint is not None:
        command += ["--resume-checkpoint", str(resume_checkpoint)]
    if args.learning_rate is not None:
        command += ["--learning-rate", str(args.learning_rate)]
    if args.text_adapter_learning_rate is not None:
        command += ["--text-adapter-learning-rate", str(args.text_adapter_learning_rate)]
    return command


def plot_metrics(output: Path, rows: list[dict[str, object]], track: str) -> None:
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    specifications = (
        ("changed_ndcg10", f"{track.lower()}_changed_ndcg10.png", "Changed text-derived nDCG@10"),
        ("changed_recall100", f"{track.lower()}_changed_recall100.png", "Changed candidate Recall@100"),
        ("train_loss", f"{track.lower()}_train_loss.png", "Training loss"),
    )
    for key, filename, title in specifications:
        points = [(int(row["epoch"]), float(row[key])) for row in rows if row.get(key) not in (None, "") and math.isfinite(float(row[key]))]
        if not points:
            continue
        figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o")
        axis.set(xlabel="Epoch", ylabel=title, title=f"QCPR {track}: {title}")
        axis.grid(alpha=0.3)
        if key == "changed_recall100" and rows[0].get("recall_floor") is not None:
            axis.axhline(float(rows[0]["recall_floor"]), color="red", linestyle="--", label="acceptance floor")
            axis.legend()
        figure.tight_layout()
        figure.savefig(plots / filename)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("A0", "B"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path,
                        help="accepted A0 model checkpoint used only for B model warm start")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--text-adapter-learning-rate", type=float)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    train_manifest = args.manifest_dir / "natural_train_retrieval_manifest.jsonl"
    validation_manifest = args.manifest_dir / "natural_validation_retrieval_manifest.jsonl"
    rows = read_rows(train_manifest)
    steps_per_epoch = len(CappedCompositionalBatchSampler(rows, args.batch_size, seed=args.seed))
    if args.batch_size != 32:
        raise ValueError("A0/B scientific protocol requires physical batch 32")

    state = {
        "track": args.track,
        "mask_pixels_used": False,
        "target_aware_crop": False,
        "physical_batch": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256(train_manifest),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": sha256(validation_manifest),
        "parent_checkpoint": None if args.parent_checkpoint is None else str(args.parent_checkpoint),
        "parent_checkpoint_sha256": None if args.parent_checkpoint is None else sha256(args.parent_checkpoint),
        "initialization_checkpoint": None if args.initialization_checkpoint is None else str(args.initialization_checkpoint),
        "initialization_checkpoint_sha256": None if args.initialization_checkpoint is None else sha256(args.initialization_checkpoint),
    }
    (args.output_dir / "run_contract.json").write_text(json.dumps(state, indent=2) + "\n")
    (args.output_dir / "environment.json").write_text(json.dumps({
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")

    parent = args.parent_checkpoint
    baseline_report = args.baseline_report
    history: list[dict[str, object]] = []
    best_value = -math.inf
    best_feasible = -math.inf
    stale = 0
    recall_floor: float | None = None
    if args.prior_run is not None:
        history = json.loads((args.prior_run / "metrics_by_epoch.json").read_text())
        feasible_rows = [row for row in history if bool(row.get("feasible"))]
        if history:
            best_value = max(float(row["changed_ndcg10"]) for row in history)
        if feasible_rows:
            best_feasible = max(float(row["changed_ndcg10"]) for row in feasible_rows)
            shutil.copy2(args.prior_run / "best_feasible.pt", args.output_dir / "best_feasible.pt")
            shutil.copy2(args.prior_run / "best_ndcg.pt", args.output_dir / "best_ndcg.pt")
            prior_development = args.prior_run / "best_feasible_development.json"
            if prior_development.exists():
                shutil.copy2(prior_development, args.output_dir / "best_feasible_development.json")
        recall_floor = float(history[0]["recall_floor"]) if history else None

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch:03d}"
        epoch_dir.parent.mkdir(parents=True, exist_ok=True)
        total_steps = epoch * steps_per_epoch
        phase = "global_bootstrap" if args.track == "A0" else "late_interaction"
        command = build_training_command(
            args, epoch=epoch, total_steps=total_steps, phase=phase,
            resume_checkpoint=parent,
        )
        run(command)

        checkpoint = epoch_dir / "last.pt"
        report_path = epoch_dir / "development.json"
        if args.track == "A0" and baseline_report is None:
            baseline_report = args.output_dir / "development_step0.json"
            run([
                sys.executable, "scripts/evaluate_qcpr_v3_fast.py", "--checkpoint", str(epoch_dir / "initial.pt"),
                "--training-report", str(epoch_dir / "smoke_report.json"), "--phase", phase,
                "--output", str(baseline_report), "--manifest", str(validation_manifest),
                "--batch-size", str(args.batch_size), "--fast-pairs", "0",
            ], allow_gate_failure=True)
        evaluate = [
            sys.executable, "scripts/evaluate_qcpr_v3_fast.py", "--checkpoint", str(checkpoint),
            "--training-report", str(epoch_dir / "smoke_report.json"), "--phase", phase,
            "--output", str(report_path), "--manifest", str(validation_manifest),
            "--batch-size", str(args.batch_size), "--fast-pairs", "0",
        ]
        if args.track == "A0" and baseline_report is not None:
            evaluate += ["--baseline-output", str(baseline_report)]
        run(evaluate, allow_gate_failure=True)
        report = json.loads(report_path.read_text())
        training = json.loads((epoch_dir / "smoke_report.json").read_text())
        losses = [float(row["loss"]) for row in training.get("history", [])]
        changed_path = "changed_only" if args.track == "A0" else "changed_only_reranked"
        changed_ndcg = metric(report, "metrics", changed_path, "semantic_ndcg_at_10")
        changed_recall = metric(report, "metrics", "changed_only", "candidate_recall_at_100")
        if recall_floor is None:
            base = json.loads(baseline_report.read_text())
            recall_floor = metric(base, "metrics", "changed_only", "candidate_recall_at_100") - 0.01
        feasible = math.isfinite(changed_recall) and changed_recall >= recall_floor
        row = {
            "epoch": epoch, "global_step": total_steps, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "train_loss": sum(losses) / max(len(losses), 1),
            "changed_ndcg10": changed_ndcg, "changed_recall100": changed_recall,
            "recall_floor": recall_floor, "feasible": feasible,
            "overall_ndcg10": metric(report, "metrics", "global_semantic_ndcg10"),
            "no_change_recall100": metric(report, "metrics", "no_change", "candidate_recall_at_100"),
        }
        history.append(row)
        write_metrics(args.output_dir / f"{args.track.lower()}_metrics_by_epoch.csv", history)
        (args.output_dir / "metrics_by_epoch.json").write_text(json.dumps(history, indent=2) + "\n")
        shutil.copy2(checkpoint, args.output_dir / "last.pt")
        if changed_ndcg > best_value:
            best_value = changed_ndcg
            shutil.copy2(checkpoint, args.output_dir / "best_ndcg.pt")
        if feasible and changed_ndcg > best_feasible:
            best_feasible = changed_ndcg
            stale = 0
            shutil.copy2(checkpoint, args.output_dir / "best_feasible.pt")
            shutil.copy2(report_path, args.output_dir / "best_feasible_development.json")
        else:
            stale += 1
        parent = checkpoint
        plot_metrics(args.output_dir, history, args.track)
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    status = (
        "ENGINEERING_PASS"
        if args.engineering_smoke and history and all(math.isfinite(float(row["train_loss"])) for row in history)
        else "PASS" if (args.output_dir / "best_feasible.pt").exists() else "HOLD"
    )
    (args.output_dir / "summary.json").write_text(json.dumps({**state, "status": status, "epochs": history}, indent=2) + "\n")
    return 0 if status in {"PASS", "ENGINEERING_PASS"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
