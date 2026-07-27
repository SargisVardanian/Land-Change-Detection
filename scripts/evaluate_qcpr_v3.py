#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from land_change_detection.models.qcpr_v3_evaluation import soft_segmentation_metrics, two_stage_retrieval_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate canonical QCPR v3 score artifacts")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=False)
    required = {"global_scores", "reranked_scores", "semantic_relevance"}
    missing = required - artifact.keys()
    if missing: raise ValueError(f"missing canonical evaluation tensors: {sorted(missing)}")
    report = two_stage_retrieval_metrics(
        artifact["global_scores"], artifact["reranked_scores"], artifact["semantic_relevance"], top_n=args.top_n,
    )
    if "mask_logits" in artifact and "mask_targets" in artifact:
        report.update(soft_segmentation_metrics(artifact["mask_logits"], artifact["mask_targets"]))
    report.update(schema_version="qcpr-v3-evaluation-v1", top_n=args.top_n, score_artifact=str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
