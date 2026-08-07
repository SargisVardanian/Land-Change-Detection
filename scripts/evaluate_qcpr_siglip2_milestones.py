#!/usr/bin/env python3
"""Run full-gallery evaluation for every saved SigLIP-2 phase milestone.

Training and evaluation are deliberately separate processes.  This keeps the
training allocation from retaining a second model/evaluator graph while still
making every required milestone an auditable ranking artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from qcpr_siglip2.training.milestones import milestone_evaluation_path, required_milestones


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS" or path.name.startswith("slurm-"):
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-run-root", required=True)
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--gallery-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--rerank-query-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    phase_root = Path(args.phase_run_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        manifest_path = phase_root / "milestone_evaluation_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        phase = str(manifest.get("phase", ""))
        expected_steps = list(required_milestones(phase))
        if manifest.get("required_milestones") != expected_steps:
            raise RuntimeError("MILESTONE_MANIFEST_CONTRACT_MISMATCH")
        records = manifest.get("milestones")
        if not isinstance(records, list) or [int(item["step"]) for item in records] != expected_steps:
            raise RuntimeError("MILESTONE_RECORD_CONTRACT_MISMATCH")

        evaluator = Path(__file__).with_name("evaluate_qcpr_siglip2_common_gallery.py")
        if not evaluator.is_file():
            raise FileNotFoundError(evaluator)
        summaries = []
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["PYTHONPATH"] = f"{evaluator.parents[1] / 'src'}:{evaluator.parents[1] / 'scripts'}"
        for record in records:
            step = int(record["step"])
            checkpoint = Path(str(record["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if sha256(checkpoint) != str(record["checkpoint_sha256"]):
                raise RuntimeError(f"MILESTONE_CHECKPOINT_SHA_MISMATCH:{step}")
            evaluation_dir = Path(str(record["evaluation_dir"]))
            expected_evaluation_dir = milestone_evaluation_path(phase_root, step)
            if evaluation_dir != expected_evaluation_dir:
                raise RuntimeError(f"MILESTONE_EVALUATION_PATH_MISMATCH:{step}")
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(evaluator),
                "--siglip2-model",
                args.siglip2_model,
                "--data-release",
                args.data_release,
                "--development-manifest",
                args.development_manifest,
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(evaluation_dir),
                "--expected-code-sha",
                args.expected_code_sha,
                "--gallery-batch-size",
                str(args.gallery_batch_size),
                "--query-batch-size",
                str(args.query_batch_size),
                "--rerank-query-batch-size",
                str(args.rerank_query_batch_size),
                "--device",
                args.device,
            ]
            subprocess.run(command, cwd=evaluator.parents[1], env=environment, check=True)
            required = [
                evaluation_dir / "evaluation_metrics.json",
                evaluation_dir / "full_rankings.pt",
                evaluation_dir / "rankings_top100.jsonl",
                evaluation_dir / "ranking_integrity.json",
                evaluation_dir / "SHA256SUMS",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError("MILESTONE_EVALUATION_ARTIFACT_MISSING:" + ",".join(missing))
            integrity = json.loads((evaluation_dir / "ranking_integrity.json").read_text())
            if not integrity.get("scores_finite") or not integrity.get("mask_free"):
                raise RuntimeError(f"MILESTONE_EVALUATION_INTEGRITY_FAILURE:{step}")
            summaries.append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "evaluation_dir": str(evaluation_dir),
                    "evaluation_metrics_sha256": sha256(evaluation_dir / "evaluation_metrics.json"),
                    "rankings_sha256": sha256(evaluation_dir / "full_rankings.pt"),
                    "ranking_integrity_sha256": sha256(evaluation_dir / "ranking_integrity.json"),
                    "status": "PASS",
                }
            )
        write_json(
            output_root / "milestone_evaluation_summary.json",
            {
                "phase": phase,
                "required_milestones": expected_steps,
                "milestones": summaries,
                "status": "PASS",
            },
        )
        write_sha256sums(output_root)
        return 0
    except Exception as exc:
        write_json(
            output_root / "failure.json",
            {
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        write_sha256sums(output_root)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
