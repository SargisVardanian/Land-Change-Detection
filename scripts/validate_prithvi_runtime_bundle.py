from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export the Prithvi runtime bundle contract.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--terratorch-config-path")
    parser.add_argument("--model-dir", default="artifacts/models/semantic/Prithvi-EO-2.0-300M-TL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from land_change_detection.training.prithvi_experimental import build_prithvi_runtime_bundle

    bundle = build_prithvi_runtime_bundle(
        checkpoint_path=args.checkpoint_path,
        terratorch_config_path=args.terratorch_config_path,
        model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(bundle.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
