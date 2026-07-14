from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.models.qcpr_v3_data import WeightingConfig, derive_qcpr_v3_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--balanced-validation-per-cell", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = derive_qcpr_v3_manifests(
        args.manifest,
        args.output_dir,
        config=WeightingConfig(args.alpha, args.tau, args.min_weight, args.max_weight),
        balanced_validation_per_cell=args.balanced_validation_per_cell,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
