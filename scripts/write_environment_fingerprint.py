from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from land_change_detection.run_metadata import jsonl_fingerprint, path_fingerprint, safe_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility fingerprint for the active retrieval environment.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--remoteclip-checkpoint", type=Path, default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--cli", default=None)
    return parser.parse_args()


def _safe_module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _nvidia_driver() -> str | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def main() -> int:
    args = parse_args()
    config_payload = None
    if args.config_json is not None and args.config_json.exists():
        config_payload = json.loads(args.config_json.read_text(encoding="utf-8"))
    payload = {
        "git_commit": safe_git_commit(args.project_root),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _safe_module_version("torchvision"),
        "cuda_build": torch.version.cuda,
        "nvidia_driver": _nvidia_driver(),
        "transformers": _safe_module_version("transformers"),
        "open_clip": _safe_module_version("open_clip"),
        "dinov2_checkpoint": path_fingerprint(args.dinov2_model_path),
        "remoteclip_checkpoint": path_fingerprint(args.remoteclip_checkpoint),
        "dataset_manifest": jsonl_fingerprint(args.dataset_manifest),
        "exact_cli": args.cli,
        "config": config_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
