#!/usr/bin/env python3
"""Recompute the frozen-model D0/D1 data-only comparison on final r18.

The ranking tensor is an already completed frozen-model evaluation.  This
utility only restricts its immutable full gallery to the final r18 gallery and
applies either the legacy exact relevance rule (D0) or the final sparse
collision-ignore rule (D1).  It never trains or changes model weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/mnt/weka/svardanyan/rs_change_project")
RELEASE = ROOT / "manifests/qcpr_bitemporal_v2_train_20260808_final_r18"
CORE = ROOT / "manifests/qcpr_bitemporal_core_benchmark_v1_r18"
RANKING_RUN = ROOT / "runs/qcpr_temporal_siglip_common_eval_stage_a_reval_eb96463_20260808"
OLD_MANIFEST = ROOT / "runs/qcpr_stage2_review_system_465e894_20260803/common_frozen_evaluation/common_exact_development.jsonl"
OUT = RELEASE / "audits/data_only_comparison_final_r18.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_query_id(value: str) -> str:
    value = str(value)
    if value.endswith(":canonical"):
        value = value[:-10]
    parts = value.split(":")
    if len(parts) > 1 and parts[0] == parts[1]:
        value = ":".join(parts[1:])
    return value


def pair_id(row: dict) -> str:
    values = row.get("positive_pair_ids") or row.get("positive_item_ids")
    if values:
        return str(values[0])
    return str(row.get("canonical_pair_id") or row.get("source_item_id"))


def metric_arrays(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "r1": float(np.mean(ranks <= 1)),
        "r5": float(np.mean(ranks <= 5)),
        "r10": float(np.mean(ranks <= 10)),
        "r50": float(np.mean(ranks <= 50)),
        "r100": float(np.mean(ranks <= 100)),
        "nDCG10": float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0))),
    }


def bootstrap_delta(d0: np.ndarray, d1: np.ndarray, seed: int = 20260808, samples: int = 10000) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(d0)
    metrics = {key: [] for key in metric_arrays(d0)}
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        a = metric_arrays(d0[indices])
        b = metric_arrays(d1[indices])
        for key in metrics:
            metrics[key].append(b[key] - a[key])
    return {
        key: {
            "delta_mean": float(metric_arrays(d1)[key] - metric_arrays(d0)[key]),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        }
        for key, values in metrics.items()
    }


def main() -> int:
    final_rows = load_jsonl(RELEASE / "manifests/exact_development.jsonl")
    final_rel = load_jsonl(CORE / "final/final_exact_development_relevance.jsonl")
    old_rows = load_jsonl(OLD_MANIFEST)
    old_pairs: list[str] = []
    for row in old_rows:
        value = pair_id(row)
        if value not in old_pairs:
            old_pairs.append(value)
    rankings = torch.load(RANKING_RUN / "full_rankings.pt", map_location="cpu", weights_only=False)
    top_rows = load_jsonl(RANKING_RUN / "rankings_top100.jsonl")
    if tuple(rankings.shape) != (len(top_rows), len(old_pairs)):
        raise RuntimeError(f"ranking shape/order mismatch: {tuple(rankings.shape)} rows={len(top_rows)} pairs={len(old_pairs)}")
    query_to_row = {normalize_query_id(row["query_id"]): index for index, row in enumerate(top_rows)}
    final_gallery = []
    for row in final_rows:
        item = str(row["positive_item_ids"][0])
        if item not in final_gallery:
            final_gallery.append(item)
    old_pair_index = {item: index for index, item in enumerate(old_pairs)}
    if not set(final_gallery).issubset(old_pair_index):
        raise RuntimeError("final gallery is not a subset of the frozen ranking gallery")
    final_indices = {old_pair_index[item] for item in final_gallery}
    relevance_by_query = {str(row["query_id"]): row for row in final_rel}
    d0_ranks: list[int] = []
    d1_ranks: list[int] = []
    source: list[str] = []
    query_order: list[str] = []
    missing: list[str] = []
    for row in final_rows:
        query_id = str(row["query_id"])
        normalized = normalize_query_id(query_id)
        if normalized not in query_to_row:
            missing.append(query_id)
            continue
        positive = {str(value) for value in row.get("positive_item_ids", [])}
        ignored = {str(value) for value in relevance_by_query[query_id].get("ignored_item_ids", [])}
        restricted = [index for index in rankings[query_to_row[normalized]].tolist() if int(index) in final_indices]
        d0_order = [old_pairs[index] for index in restricted]
        d1_order = [item for item in d0_order if item not in ignored]
        d0_rank = next((index + 1 for index, item in enumerate(d0_order) if item in positive), None)
        d1_rank = next((index + 1 for index, item in enumerate(d1_order) if item in positive), None)
        if d0_rank is None or d1_rank is None:
            raise RuntimeError(f"positive missing from restricted ranking for {query_id}")
        d0_ranks.append(d0_rank)
        d1_ranks.append(d1_rank)
        source.append(str(row["source_item_id"]).split(":", 1)[0])
        query_order.append(query_id)
    if missing:
        raise RuntimeError(f"missing {len(missing)} final queries from frozen ranking run")
    d0 = np.asarray(d0_ranks, dtype=np.float64)
    d1 = np.asarray(d1_ranks, dtype=np.float64)
    d0_metrics = metric_arrays(d0)
    d1_metrics = metric_arrays(d1)
    per_source = {}
    for name in sorted(set(source)):
        mask = np.asarray([value == name for value in source])
        per_source[name] = {"query_count": int(mask.sum()), "D0": metric_arrays(d0[mask]), "D1": metric_arrays(d1[mask])}
    output = {
        "schema_version": "qcpr-final-r18-data-only-comparison-v1",
        "status": "PASS_FINAL_R18_MATCHED_FROZEN_MODEL_DATA_ONLY",
        "improvement_claim": False,
        "scientific_claim": "NO_MODEL_IMPROVEMENT_CLAIM; D0/D1 use the same frozen ranking tensor and identical final r18 gallery/query pixels. Any delta is the relevance-policy collision-ignore effect.",
        "D0": {"description": "final r18 exact gallery with legacy single-positive relevance; no sparse collision ignore", "metrics": d0_metrics},
        "D1": {"description": "final r18 exact gallery with sparse collision/ambiguous IGNORE relevance policy", "metrics": d1_metrics},
        "paired_delta_D1_minus_D0": bootstrap_delta(d0, d1),
        "per_source": per_source,
        "matched_contract": {
            "query_count": len(query_order),
            "gallery_pair_count": len(final_gallery),
            "query_order_sha256": hashlib.sha256("\n".join(query_order).encode()).hexdigest(),
            "gallery_order_sha256": hashlib.sha256("\n".join(final_gallery).encode()).hexdigest(),
            "final_development_manifest": str(RELEASE / "manifests/exact_development.jsonl"),
            "final_development_manifest_sha256": sha256_file(RELEASE / "manifests/exact_development.jsonl"),
            "final_relevance_sha256": sha256_file(CORE / "final/final_exact_development_relevance.jsonl"),
            "source_frozen_gallery_count": len(old_pairs),
            "restricted_from_source_frozen_gallery": True,
            "query_order_identical_to_final_manifest": query_order == [str(row["query_id"]) for row in final_rows],
        },
        "frozen_model": {
            "ranking_run": str(RANKING_RUN),
            "ranking_tensor_sha256": sha256_file(RANKING_RUN / "full_rankings.pt"),
            "ranking_integrity_sha256": sha256_file(RANKING_RUN / "retrieval_integrity_audit.json"),
            "model_source": json.loads((RANKING_RUN / "model_source.json").read_text(encoding="utf-8")),
            "code_state": json.loads((RANKING_RUN / "code_state.json").read_text(encoding="utf-8")),
            "training_launched_by_this_comparison": False,
        },
        "D2": {"status": "HOLD", "reason": "verified Forest/RSCC text absent"},
        "D3": {"status": "HOLD", "reason": "verified localized/long-series text absent"},
        "semantic_metrics": {"status": "NOT_RUN", "reason": "immutable human semantic judgments absent"},
        "stable_scene_metrics": {"status": "NOT_RUN", "reason": "stable human gate absent"},
        "localized_metrics": {"status": "NOT_RUN", "reason": "verified localized language absent"},
        "long_series_metrics": {"status": "NOT_RUN", "reason": "TAMMs license/temporal review absent"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mirror = RELEASE / "source_reports/data_only_comparison_final_r18.json"
    mirror.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "sha256": sha256_file(OUT), "query_count": len(query_order), "gallery_pair_count": len(final_gallery), "D0": d0_metrics, "D1": d1_metrics, "paired_delta": output["paired_delta_D1_minus_D0"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
