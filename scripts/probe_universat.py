from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.run_metadata import path_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline probe for UniverSat joint temporal backend.")
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "external" / "UniverSat")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "models" / "universat-base")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "model_probes" / "universat.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "repo_id": "g-astruc/UniverSat",
        "source_commit": "f6df2eec54955b0f7524cc95fe21a5e80c0239d9",
        "source_dir": str(args.source_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "mode": "universat_joint_public",
        "status": "not_started",
    }
    try:
        backend = UniverSatJointBackend(
            UniverSatBackendConfig(source_dir=args.source_dir, checkpoint_dir=args.checkpoint_dir)
        ).to(args.device)
        t1 = torch.rand(1, 3, args.image_size, args.image_size, device=args.device)
        t2 = torch.rand(1, 3, args.image_size, args.image_size, device=args.device)
        with torch.no_grad():
            features = backend(t1, t2)
        report.update(
            {
                "status": "ok",
                "parameter_count": backend.model_parameter_count,
                "dtype": backend.dtype,
                "source_fingerprint": path_fingerprint(args.source_dir),
                "checkpoint_fingerprint": path_fingerprint(args.checkpoint_dir),
                "global_shape": list(features.global_embedding.shape),
                "local_shape": list(features.local_tokens.shape),
                "grid": [features.grid_height, features.grid_width],
                "metadata": features.metadata,
            }
        )
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
