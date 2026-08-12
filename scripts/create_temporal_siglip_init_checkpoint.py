#!/usr/bin/env python3
"""Create the deterministic clean-pretrained TemporalSigLIP step-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from qcpr_temporal_siglip.backbone import TemporalSigLIPBackbone
from qcpr_temporal_siglip.config import TemporalSigLIPConfig
from qcpr_temporal_siglip.model import TemporalSigLIP


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_state(expected_sha: str) -> dict[str, Any]:
    worktree = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    clean = not bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    if head != expected_sha or not clean:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    return {"head": head, "worktree_clean": clean}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TEMPORAL_SIGLIP_INIT_REQUIRES_CUDA")
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    state = code_state(args.expected_code_sha)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    config = TemporalSigLIPConfig().validate()
    backbone = TemporalSigLIPBackbone(
        args.siglip2_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model = TemporalSigLIP(backbone, config).to(device)
    model.set_stage_a()
    checkpoint = run / "checkpoint_step_0.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "global_step": 0,
            "metadata": {
                "phase": "A",
                "code_sha": args.expected_code_sha,
                "seed": args.seed,
                "status": "CLEAN_PRETRAINED_STEP_ZERO",
            },
        },
        checkpoint,
    )
    write_json(run / "code_state.json", state)
    write_json(run / "model_contract.json", {
        "model": "TemporalSigLIP",
        "config": config.to_dict(),
        "step": 0,
        "status": "CLEAN_PRETRAINED_STEP_ZERO",
    })
    write_json(run / "checkpoint.json", {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "global_step": 0,
        "seed": args.seed,
    })
    write_json(run / "environment.json", {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
