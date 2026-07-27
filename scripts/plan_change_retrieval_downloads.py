from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.data.dataset_registry import build_download_plan, render_human_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan change-retrieval downloads under a storage budget.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "manual", "deferred", "all"), default="baseline")
    parser.add_argument("--reserve-gb", type=float, default=120.0)
    parser.add_argument("--max-download-gb", type=float, default=50.0)
    parser.add_argument("--allow-unknown-size", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_download_plan(
        args.project_root,
        phase=args.phase,
        reserve_gb=args.reserve_gb,
        max_download_gb=args.max_download_gb,
        allow_unknown_size=args.allow_unknown_size,
    )
    print(render_human_plan(plan))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 1 if plan["policy_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
