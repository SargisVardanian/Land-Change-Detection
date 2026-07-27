#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-root", type=Path, required=True)
    args = parser.parse_args()
    a0 = json.loads((args.chain_root / "a0_long/summary.json").read_text())
    b = json.loads((args.chain_root / "b_long/summary.json").read_text())
    c0 = json.loads((args.chain_root / "c0_development/c0_emergent_metrics.json").read_text())
    checkpoints = {}
    for name in ("a0_long/best_feasible.pt", "a0_long/best_ndcg.pt", "b_long/best_feasible.pt", "b_long/best_ndcg.pt"):
        path = args.chain_root / name
        if path.exists():
            checkpoints[name] = {"path": str(path), "sha256": sha256(path)}
    summary = {
        "a0_long_training": a0["status"],
        "b_local_retrieval": b["status"],
        "c0_emergent_localization": "DEMONSTRATED" if c0.get("localization_margin", 0) > 0 and c0.get("soft_dice", 0) > 0 else "WEAK_SIGNAL",
        "s0_supervised_probe": "SEPARATE_ABLATION",
        "untouched_retrieval_test": "NOT_EVALUATED",
        "untouched_mask_test": "NOT_EVALUATED",
        "checkpoints": checkpoints,
    }
    (args.chain_root / "checkpoint_inventory.json").write_text(
        json.dumps(checkpoints, indent=2) + "\n"
    )
    (args.chain_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# QCPR mask-free emergent grounding long run", "",
        "## Status", "",
        f"- A0: **{summary['a0_long_training']}**",
        f"- B: **{summary['b_local_retrieval']}**",
        f"- C0: **{summary['c0_emergent_localization']}**",
        "- S0: **SEPARATE_ABLATION**", "",
        "C0 is evaluated only after B is frozen and is derived from canonical token–patch similarities. No S2Looking mask pixels were used by A0 or B training.", "",
        "## Evidence boundaries", "",
        "Retrieval relevance is text-derived pseudo-relevance unless explicitly human-reviewed. S2Looking development masks are used only in the post-training C0 evaluator. Untouched tests were not used.", "",
        "## Artifacts", "",
    ]
    lines.extend(f"- `{name}` — `{entry['sha256']}`" for name, entry in checkpoints.items())
    (args.chain_root / "report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
