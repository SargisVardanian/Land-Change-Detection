#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    candidates = []
    for name in ("A", "B", "C"):
        path = root / name / "smoke_report.json"
        report = json.loads(path.read_text())
        history = report["fixed_validation_history"]
        final = history[-1]
        candidates.append({
            "objective": name,
            "report": str(path),
            "runtime_status": report["runtime_status"],
            "gradient_status": report["gradient_status"],
            "scientific_status": report["scientific_status"],
            "micro_overfit_passed": bool(report["micro_overfit_gate"]["passed"]),
            "validation": final,
        })
    eligible = [row for row in candidates if row["micro_overfit_passed"]]
    eligible.sort(
        key=lambda row: (
            row["validation"]["nonempty_dice"],
            row["validation"]["nonempty_iou"],
            row["validation"]["nonempty_precision"],
            row["validation"]["foreground_background_margin"],
            -row["validation"]["empty_false_positive_area"],
        ),
        reverse=True,
    )
    return {
        "status": "MICRO_OVERFIT_PASS" if eligible else "SCIENTIFIC_HOLD",
        "selection_rule": "validation Dice, IoU, precision, FG-BG margin, then lower empty FP; training loss excluded",
        "selected_objective": eligible[0]["objective"] if eligible else None,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
