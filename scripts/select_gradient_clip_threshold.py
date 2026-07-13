from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def select_threshold(rows: list[dict], target_clipping_fraction: float = 0.2) -> dict:
    if not 0.0 < target_clipping_fraction < 1.0:
        raise ValueError("target_clipping_fraction must be between zero and one")
    norms = [
        float(row["grad_norm_before_clip"])
        for row in rows
        if row.get("record_type") in (None, "train_step")
        and "grad_norm_before_clip" in row
        and math.isfinite(float(row["grad_norm_before_clip"]))
    ]
    if not norms:
        raise ValueError("No finite train-step gradient norms found")
    values = torch.tensor(norms, dtype=torch.float64)
    threshold = float(torch.quantile(values, 1.0 - target_clipping_fraction).item())
    module_keys = sorted({key for row in rows for key in row if key.startswith("grad_norm_") and key != "grad_norm_before_clip"})
    modules = {}
    for key in module_keys:
        module_values = torch.tensor([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))], dtype=torch.float64)
        if module_values.numel():
            modules[key.removeprefix("grad_norm_")] = {
                "count": int(module_values.numel()),
                "p50": float(torch.quantile(module_values, 0.5).item()),
                "p80": float(torch.quantile(module_values, 0.8).item()),
                "p90": float(torch.quantile(module_values, 0.9).item()),
                "max": float(module_values.max().item()),
            }
    return {
        "status": "PASS",
        "step_count": len(norms),
        "target_clipping_fraction": target_clipping_fraction,
        "suggested_grad_clip_norm": threshold,
        "resulting_clipping_fraction": float((values > threshold).double().mean().item()),
        "global_norm": {
            "p50": float(torch.quantile(values, 0.5).item()),
            "p80": float(torch.quantile(values, 0.8).item()),
            "p90": float(torch.quantile(values, 0.9).item()),
            "max": float(values.max().item()),
        },
        "module_norms": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select an evidence-based gradient clipping threshold.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-clipping-fraction", type=float, default=0.2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.history.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = select_threshold(rows, args.target_clipping_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
