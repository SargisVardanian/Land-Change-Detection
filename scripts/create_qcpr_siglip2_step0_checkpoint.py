#!/usr/bin/env python3
"""Create a deterministic frozen SigLIP-2 step-0 checkpoint.

The checkpoint contains the pinned pretrained towers plus the freshly
initialized temporal/evidence heads.  No data is loaded and no optimizer step
is performed; it is the immutable initialization used by common-gallery
evaluation and Phase-A comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_state(worktree: Path) -> dict[str, Any]:
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "-C", str(worktree), "branch", "--show-current"], text=True
        ).strip(),
        "worktree_clean": not bool(
            subprocess.check_output(
                ["git", "-C", str(worktree), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = Path(args.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    worktree = Path(__file__).resolve().parents[1]
    state = git_state(worktree)
    if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    model_path = Path(args.siglip2_model)
    config_path = Path(args.config_path)
    if not model_path.is_dir() or not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(model_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config must contain a JSON object")
    fields = {
        name: payload[name]
        for name in Siglip2TemporalConfig.__dataclass_fields__
        if name in payload
    }
    config = Siglip2TemporalConfig.from_dict(fields)
    torch.manual_seed(args.seed)
    backbone = Siglip2Backbone(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    )
    model = Siglip2TemporalRetrievalModel(backbone, config)
    model.eval()
    checkpoint = run / "checkpoint_step_0.pt"
    checkpoint_payload = {
        "model_state": model.state_dict(),
        "global_step": 0,
        "metadata": {
            "code_sha": args.expected_code_sha,
            "phase": "FROZEN_STEP_0",
            "optimizer_resumed": False,
            "seed": args.seed,
        },
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    (run / "checkpoint.sha256").write_text(
        f"{checkpoint_hash}  {checkpoint.name}\n", encoding="utf-8"
    )
    write_json(run / "code_state.json", {"expected_code_sha": args.expected_code_sha, "runtime_git": state})
    write_json(
        run / "model_source.json",
        {
            "repository": "google/siglip2-base-patch16-256",
            "local_path": str(model_path),
            "weights_sha256": sha256(model_path / "model.safetensors"),
            "runtime_class": backbone.runtime_class,
            "parameter_scope": backbone.parameter_scope_report(),
        },
    )
    write_json(
        run / "resolved_config.json",
        {"phase": "FROZEN_STEP_0", "seed": args.seed, "config": config.to_dict()},
    )
    write_json(
        run / "training_complete.json",
        {
            "status": "PASS",
            "phase": "FROZEN_STEP_0",
            "global_step": 0,
            "checkpoint": str(checkpoint),
            "optimizer_steps": 0,
            "scientific_interpretation": "initialization checkpoint only",
        },
    )
    lines = []
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.name}")
    (run / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
