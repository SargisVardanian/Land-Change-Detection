#!/usr/bin/env python3
"""Audit existing B1/P1 artifacts without inferring training benefit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def compact_metrics(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state.get("status"),
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "development_manifest_sha256": state.get("development_manifest_sha256"),
        "full_rankings_sha256": state.get("full_rankings_sha256"),
        "physical_pair_count": state.get("physical_pair_count"),
        "query_count": state.get("query_count"),
        "metrics": state.get("metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--p1-root", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_state = read_json(args.baseline_root / "common_frozen_evaluation_state.json")
    p1_state = read_json(args.p1_root / "common_frozen_evaluation_state.json")
    artifacts: dict[str, dict[str, Any]] = {}
    for name, root in (("baseline", args.baseline_root), ("p1", args.p1_root)):
        for relative in ("common_frozen_evaluation_state.json", "evaluation_summary.json", "full_rankings.pt", "rankings_top100.jsonl"):
            path = root / relative
            artifacts[f"{name}:{relative}"] = {
                "exists": path.exists(),
                "sha256": sha256(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }

    comparison = None
    if args.comparison_json and args.comparison_json.exists():
        comparison = read_json(args.comparison_json)
    p1_control = args.p1_root.parent / "qcpr_stage2_p1_control_47a628f_20260804" / "p1"
    feature_cache_meta = p1_control / "feature_cache" / "development" / "cache_meta.json"
    report: dict[str, Any] = {
        "status": "PASS_ARTIFACT_AUDIT",
        "comparison_status": "COMPLETE_ARTIFACT" if comparison is not None else "INCOMPLETE_ARTIFACT",
        "baseline": compact_metrics(baseline_state),
        "p1": compact_metrics(p1_state),
        "comparison_artifact": str(args.comparison_json) if comparison is not None else None,
        "comparison": comparison,
        "artifact_inventory": artifacts,
        "feature_cache": {
            "metadata_exists": feature_cache_meta.exists(),
            "metadata_sha256": sha256(feature_cache_meta) if feature_cache_meta.exists() else None,
            "interpretation": "P1 used a frozen feature cache; no backbone adaptation is demonstrated by these artifacts.",
        },
        "diagnostic_limits": {
            "embedding_effective_rank": "NOT_AVAILABLE_FROM_RANKING_ONLY",
            "embedding_norm_distribution": "NOT_AVAILABLE_FROM_RANKING_ONLY",
            "source_classification_separability": "NOT_AVAILABLE_FROM_RANKING_ONLY",
            "gradient_norms": "NOT_AVAILABLE_FROM_EVALUATION_ARTIFACTS",
            "candidate_recall": {
                "baseline": {str(k): baseline_state.get("metrics", {}).get("exact_primary", {}).get(f"recall_at_{k}") for k in (10, 50, 100, 500)},
                "p1": {str(k): p1_state.get("metrics", {}).get("exact_primary", {}).get(f"recall_at_{k}") for k in (10, 50, 100, 500)},
            },
        },
        "scientific_conclusion": (
            "Do not claim P1 improvement. The verified common-gallery artifact must be used; "
            "feature-cache and ranking-only diagnostics do not establish backbone adaptation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = [
        "# QCPR v3 B1/P1 diagnostic audit",
        "",
        f"- status: {report['status']}",
        f"- comparison status: {report['comparison_status']}",
        f"- baseline root: {args.baseline_root}",
        f"- P1 root: {args.p1_root}",
        "",
        "The report preserves the common-gallery metrics and explicitly marks diagnostics unavailable when the "
        "stored artifacts do not contain embeddings or gradients. It does not infer a training improvement from loss.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
