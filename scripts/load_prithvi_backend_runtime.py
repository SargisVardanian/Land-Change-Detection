from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and load the PrithviTerratorchBackend runtime scaffold.")
    parser.add_argument("--model-dir", default="artifacts/models/semantic/Prithvi-EO-2.0-300M-TL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from land_change_detection.segmentation_backends.prithvi_terratorch import PrithviTerratorchBackend

    backend = PrithviTerratorchBackend(model_dir=args.model_dir, device=args.device)
    payload = backend.load_runtime_object().to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
