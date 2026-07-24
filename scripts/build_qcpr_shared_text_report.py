#!/usr/bin/env python3
"""Report only retrieval-trained A0/B and emergent patch-evidence localization."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load(path:Path):
    return json.loads(path.read_text()) if path.exists() else None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); args=p.parse_args()
    root=args.run_root
    report={
      "contract":load(root/"run_contract.json"),
      "a0":load(root/"a0"/"summary.json"),
      "b":load(root/"b"/"summary.json"),
      "evaluation":load(root/"evaluation"/"c0_emergent_metrics.json"),
      "claim":"A0/B retrieval training only. Localization is an emergent B token-patch evidence map; no segmentation decoder was trained."
    }
    (root/"report").mkdir(exist_ok=True)
    (root/"report"/"retrieval_only_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (root/"report"/"report.md").write_text(
        "# QCPR shared-text retrieval-only report\n\n"
        "A0 and B are trained. Soft maps are direct emergent token-patch evidence, not supervised segmentation.\n"
    )
if __name__=="__main__": main()
