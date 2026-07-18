#!/usr/bin/env python3
"""Evaluator-only A0 score fusion and checkpoint feasibility audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from evaluate_qcpr_v3_fast import global_subset_metrics
from land_change_detection.models.retrieval_heads import classify_caption_semantics


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"global_scores", "relevance", "caption_to_pair", "captions", "pair_datasets", "query_groups"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"score artifact {path} is missing {missing}")
    return payload


def _metrics(scores: torch.Tensor, artifact: dict) -> dict[str, float | int]:
    changed = artifact["query_groups"]["changed_only"].bool()
    return global_subset_metrics(scores, artifact["relevance"], changed, k=100)


def _lost_rows(baseline: dict, trained: dict, output: Path) -> dict[str, object]:
    base_top = baseline["global_scores"].topk(100, dim=1).indices
    trained_top = trained["global_scores"].topk(100, dim=1).indices
    relevance = baseline["relevance"].bool()
    rows: list[dict[str, object]] = []
    for query in range(relevance.shape[0]):
        base_hit = bool(relevance[query].gather(0, base_top[query]).any())
        trained_hit = bool(relevance[query].gather(0, trained_top[query]).any())
        if base_hit and not trained_hit:
            caption = str(baseline["captions"][query])
            semantic = classify_caption_semantics(caption)
            rows.append({
                "query_index": query,
                "caption": caption,
                "dataset": str(baseline["pair_datasets"][int(baseline["caption_to_pair"][query])]),
                "caption_group": " ".join(caption.lower().split()),
                "change_type": "changed" if bool(baseline["query_groups"]["changed_only"][query]) else "no_change",
                "exact_pair_relevant": bool(relevance[query, int(baseline["caption_to_pair"][query])]),
                "semantic_relevance_lost": True,
                "semantic_tags": semantic,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {"lost_query_count": len(rows), "path": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--trained-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, trained = _load(args.baseline_artifact), _load(args.trained_artifact)
    for key in ("relevance", "caption_to_pair", "captions", "pair_datasets"):
        if baseline[key] != trained[key] if isinstance(baseline[key], list) else not torch.equal(baseline[key], trained[key]):
            raise ValueError(f"baseline/trained evaluation fingerprint mismatch for {key}")
    if baseline["global_scores"].shape != trained["global_scores"].shape:
        raise ValueError("baseline/trained score matrix shapes differ")
    baseline_changed_recall = float(_metrics(baseline["global_scores"], baseline)["candidate_recall_at_100"])
    recall_floor = baseline_changed_recall - 0.01
    rows = []
    for index in range(21):
        alpha = index / 20.0
        score = (1.0 - alpha) * baseline["global_scores"] + alpha * trained["global_scores"]
        metric = _metrics(score, baseline)
        rows.append({"alpha": alpha, **metric, "feasible": float(metric["candidate_recall_at_100"]) >= recall_floor})
    feasible = [row for row in rows if row["feasible"]]
    selected = max(feasible, key=lambda row: (float(row["semantic_ndcg_at_10"]), -float(row["alpha"]))) if feasible else None
    lost_path = args.output.with_name("a0_lost_top100.csv")
    audit = _lost_rows(baseline, trained, lost_path)
    report = {
        "schema_version": "qcpr-v3-a0-fusion-audit-v1",
        "baseline_artifact": str(args.baseline_artifact),
        "trained_artifact": str(args.trained_artifact),
        "recall_floor": recall_floor,
        "baseline_changed_recall_at_100": baseline_changed_recall,
        "sweep": rows,
        "selected_alpha": selected,
        "selected_status": "PASS" if selected is not None else "NO_FEASIBLE_ALPHA",
        "lost_top100_audit": audit,
        "checkpoint_availability": {
            "step_0": "available",
            "step_1_to_99": "not_saved_in_existing_run",
            "step_100": "available",
            "best_ndcg_checkpoint": "step_100 (last.pt)",
            "best_feasible_checkpoint": "step_0 (initial.pt)" if selected is None else "requires checkpoint-level verification",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
