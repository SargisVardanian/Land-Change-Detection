from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.levir_cc_audit import LEVIR_CC_PRESETS, all_exist, summarize_required_files, summarize_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the LEVIR-CC cluster milestone is fully evidenced.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_report(project_root: Path) -> dict:
    runs_root = project_root / "runs"
    required_files = summarize_required_files(runs_root)
    runs = summarize_runs(runs_root, LEVIR_CC_PRESETS)
    ready_presets = sorted([preset for preset, payload in runs.items() if payload["milestone_ready"]])
    report = {
        "project_root": str(project_root),
        "runs_root": str(runs_root),
        "required_files": required_files,
        "required_files_complete": all_exist(required_files),
        "runs": runs,
        "ready_presets": ready_presets,
        "milestone_ready": bool(all_exist(required_files) and ready_presets),
    }
    return report


def main() -> int:
    args = parse_args()
    report = build_report(args.project_root)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["milestone_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
