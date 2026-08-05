#!/usr/bin/env python3
"""Run a frozen-B1 data-only retrieval comparison on Dataset-v2 views.

This evaluator lives in the dataset worktree and imports the model worktree
only as a frozen runtime dependency. It never trains or writes to the model
worktree. Galleries are the union of positive physical items in each frozen
view; all metrics are query-level and use graded positive sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


def query_key(row: dict[str, Any]) -> str:
    source_item = str(row.get("source_item_id") or row.get("canonical_pair_id") or "")
    return source_item + "|" + normalize_text(str(row.get("text") or row.get("caption") or ""))


def load_items(release: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in read_jsonl(release / "registries/physical_items.jsonl")}


def load_queries(release: Path, manifest_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in manifest_names:
        rows.extend(read_jsonl(release / "manifests" / name))
    return rows


def make_view(name: str, rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        positive_ids = [str(item_id) for item_id in row.get("positive_item_ids") or []]
        if not positive_ids or any(item_id not in items for item_id in positive_ids):
            skipped += 1
            continue
        if any(str(items[item_id].get("item_type")) != "pair" for item_id in positive_ids):
            skipped += 1
            continue
        valid.append(row)
    gallery_ids = sorted({str(item_id) for row in valid for item_id in row.get("positive_item_ids") or []})
    return {
        "name": name,
        "rows": valid,
        "gallery_ids": gallery_ids,
        "query_count": len(valid),
        "gallery_pair_count": len(gallery_ids),
        "skipped_rows": skipped,
    }


def pair_rows(item_ids: list[str], items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item_id in item_ids:
        item = items[item_id]
        frames = item.get("frames") or []
        if len(frames) != 2:
            raise ValueError(f"frozen B1 evaluator requires pairs: {item_id}")
        rows.append(
            {
                "pair_id": item_id,
                "dataset_name": item.get("source", "unknown"),
                "t1_path": frames[0]["path"],
                "t2_path": frames[1]["path"],
            }
        )
    return rows


def random_retrieval_baseline(gallery_size: int, positive_size: int) -> dict[str, float]:
    if gallery_size <= 0 or positive_size <= 0:
        return {"expected_mrr_single_positive": 0.0, "expected_r1": 0.0, "expected_r5": 0.0, "expected_r10": 0.0}
    p = min(positive_size, gallery_size)
    expected = {}
    for k in (1, 5, 10):
        if k >= gallery_size:
            probability = 1.0
        else:
            probability = 1.0
            for offset in range(k):
                probability *= (gallery_size - p - offset) / (gallery_size - offset)
            probability = 1.0 - probability
        expected[f"expected_r{k}"] = probability
    harmonic = sum(1.0 / rank for rank in range(1, gallery_size + 1)) / gallery_size
    expected["expected_mrr_single_positive"] = harmonic
    return expected


def dcg(grades: list[int], cutoff: int) -> float:
    return sum((2.0 ** grade - 1.0) / math.log2(index + 2.0) for index, grade in enumerate(grades[:cutoff]))


def query_metrics(order: list[int], gallery_ids: list[str], positive_grades: dict[str, int]) -> dict[str, float]:
    ranks = {item_id: index + 1 for index, item_id in enumerate(gallery_ids[index] for index in order)}
    positive_ranks = sorted(ranks[item_id] for item_id in positive_grades if item_id in ranks)
    best_rank = positive_ranks[0] if positive_ranks else len(gallery_ids) + 1
    positive_set = set(positive_grades)
    top10 = [gallery_ids[index] for index in order[:10]]
    top5 = set(gallery_ids[index] for index in order[:5])
    top1 = set(gallery_ids[index] for index in order[:1])
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        item_id = gallery_ids[index]
        if item_id in positive_set:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / max(len(positive_set), 1)
    ranked_grades = [int(positive_grades.get(gallery_ids[index], 0)) for index in order]
    ideal_grades = sorted((int(value) for value in positive_grades.values()), reverse=True)
    return {
        "mrr": 1.0 / best_rank,
        "recall_at_1": float(bool(top1 & positive_set)),
        "recall_at_5": float(bool(top5 & positive_set)),
        "recall_at_10": float(bool(set(top10) & positive_set)),
        "average_precision": average_precision,
        "ndcg_at_10": dcg(ranked_grades, 10) / max(dcg(ideal_grades, 10), 1e-12),
        "best_positive_rank": float(best_rank),
        "positive_set_size": float(len(positive_set)),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def bootstrap_mean(values: list[float], *, rounds: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None], "rounds": 0}
    generator = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(rounds):
        samples.append(sum(values[generator.randrange(n)] for _ in range(n)) / n)
    return {
        "n": n,
        "mean": sum(values) / n,
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "rounds": rounds,
    }


def summarize_query_metrics(
    records: list[dict[str, Any]],
    gallery_size: int,
    *,
    include_per_source: bool = True,
) -> dict[str, Any]:
    if not records:
        return {"query_count": 0, "gallery_pair_count": gallery_size, "metrics": None}
    metric_names = ["mrr", "recall_at_1", "recall_at_5", "recall_at_10", "average_precision", "ndcg_at_10"]
    metrics = {name: sum(float(record[name]) for record in records) / len(records) for name in metric_names}
    source_metrics = {}
    if include_per_source:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_source[str(record["source"])].append(record)
        for source, source_records in sorted(by_source.items()):
            source_metrics[source] = summarize_query_metrics(
                source_records,
                gallery_size,
                include_per_source=False,
            )["metrics"]
    average_positive_set_size = sum(float(record["positive_set_size"]) for record in records) / len(records)
    return {
        "query_count": len(records),
        "gallery_pair_count": gallery_size,
        "metrics": metrics,
        "average_positive_set_size": average_positive_set_size,
        "random_retrieval_baseline": random_retrieval_baseline(gallery_size, round(average_positive_set_size)),
        "per_source": source_metrics,
    }


@torch.inference_mode()
def load_frozen_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    model_root = args.model_root
    sys.path[:0] = [str(model_root / "src"), str(model_root / "scripts")]
    from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
    from land_change_detection.backbones.sequence_universat import SequenceUniverSatEncoder
    from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
    from land_change_detection.backbones.universat_frame_backend import UniverSatFrameBackend
    from land_change_detection.models.qcpr_single_pass import SinglePassConfig
    from screen_qcpr_stage2_compatible_architectures import ScreenModel, image_batch, load_text_batch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = payload.get("kind") or payload.get("model_kind")
    if kind != "B1":
        raise RuntimeError(f"expected frozen B1 checkpoint, got {kind!r}")
    config_keys = set(SinglePassConfig.__dataclass_fields__)
    cfg = SinglePassConfig(**{key: value for key, value in (payload.get("config") or {}).items() if key in config_keys})
    device = torch.device("cuda", torch.cuda.current_device())
    model = ScreenModel("B1", cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    text_encoder = JinaV5TextEncoder(
        JinaV5TextConfig(model_path=args.jina_model, max_length=256, global_projection_mode="matryoshka_truncate", freeze=True)
    ).to(device).eval()
    backend = UniverSatJointBackend(
        UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint, output_grid=32, freeze=True)
    ).to(device).eval()
    frame_backend = UniverSatFrameBackend(backend.model, output_grid=32, visual_dim=768).to(device).eval()
    encoder = SequenceUniverSatEncoder(frame_backend, output_grid=32, visual_dim=768, freeze=True).to(device).eval()
    return model, text_encoder, encoder, image_batch, load_text_batch


@torch.inference_mode()
def extract_pair_embeddings(
    rows: list[dict[str, Any]],
    model: Any,
    encoder: Any,
    image_batch_fn: Any,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for start in range(0, len(rows), batch_size):
        indices = list(range(start, min(start + batch_size, len(rows))))
        batch = image_batch_fn(rows, indices, image_size).to(device)
        features = encoder(batch)
        features = features.features if hasattr(features, "features") else features
        if tuple(features.shape[1:]) != (2, 1024, 768):
            raise RuntimeError(f"unexpected frozen visual feature shape {tuple(features.shape)}")
        embeddings = model.pair_embeddings(features).detach().cpu()
        for local, index in enumerate(indices):
            result[str(rows[index]["pair_id"])] = embeddings[local]
    return result


@torch.inference_mode()
def evaluate_view(
    view: dict[str, Any],
    pair_embeddings: dict[str, torch.Tensor],
    model: Any,
    text_encoder: Any,
    load_text_batch_fn: Any,
    device: torch.device,
    text_batch: int,
    bootstrap_rounds: int,
) -> dict[str, Any]:
    gallery_ids = view["gallery_ids"]
    gallery_embeddings = torch.stack([pair_embeddings[item_id] for item_id in gallery_ids]).to(device, dtype=torch.float32)
    records: list[dict[str, Any]] = []
    for start in range(0, len(view["rows"]), text_batch):
        query_rows = view["rows"][start : start + text_batch]
        texts = [str(row.get("text") or row.get("caption") or "") for row in query_rows]
        base, tokens, attention, content = load_text_batch_fn(text_encoder, texts, device)
        text_embeddings = model._text_embedding(base, tokens, attention, content)
        scores = model.logit_scale.exp().clamp(1e-3, 100.0) * (text_embeddings @ gallery_embeddings.T) + model.logit_bias
        orders = torch.argsort(scores, dim=1, descending=True).cpu().tolist()
        for row, order in zip(query_rows, orders, strict=True):
            positive_ids = [str(item_id) for item_id in row.get("positive_item_ids") or []]
            grades = {str(item_id): int(grade) for item_id, grade in (row.get("graded_relevance") or {}).items()}
            query_result = query_metrics(order, gallery_ids, {item_id: grades[item_id] for item_id in positive_ids if item_id in grades})
            source_item = str(row.get("source_item_id") or "")
            source = str((row.get("provenance") or {}).get("source_dataset") or "unknown")
            if source == "unknown" and source_item in pair_embeddings:
                source = str(view["items"][source_item].get("source", "unknown"))
            records.append(
                {
                    **query_result,
                    "query_id": str(row.get("query_id")),
                    "query_key": query_key(row),
                    "source": source,
                    "query_scope": row.get("query_scope"),
                    "text": str(row.get("text") or row.get("caption") or ""),
                }
            )
    summary = summarize_query_metrics(records, len(gallery_ids))
    summary["view"] = view["name"]
    summary["skipped_rows"] = view["skipped_rows"]
    summary["per_query"] = records
    return summary


def paired_bootstrap(left: dict[str, Any], right: dict[str, Any], rounds: int, seed: int) -> dict[str, Any]:
    left_records = {record["query_key"]: record for record in left.get("per_query", [])}
    right_records = {record["query_key"]: record for record in right.get("per_query", [])}
    keys = sorted(set(left_records) & set(right_records))
    metrics = {}
    for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "average_precision", "ndcg_at_10"):
        deltas = [float(right_records[key][metric]) - float(left_records[key][metric]) for key in keys]
        metrics[metric] = bootstrap_mean(deltas, rounds=rounds, seed=seed + len(metric))
    return {"left_view": left["view"], "right_view": right["view"], "paired_query_count": len(keys), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-release", type=Path, required=True)
    parser.add_argument("--repaired-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("data-only comparison requires CUDA; submit through Slurm GPU allocation")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    d0_items = load_items(args.d0_release)
    repaired_items = load_items(args.repaired_release)
    all_items = dict(d0_items)
    all_items.update(repaired_items)

    d0_exact = make_view(
        "D0_current_exact_core",
        load_queries(args.d0_release, ["exact_development.jsonl"]),
        all_items,
    )
    d1_exact = make_view(
        "D1_repaired_exact",
        load_queries(args.repaired_release, ["exact_development.jsonl"]),
        all_items,
    )
    d1_stable = make_view(
        "D1_stable_scene",
        load_queries(args.repaired_release, ["stable_development.jsonl"]),
        all_items,
    )
    d1_semantic = make_view(
        "D1_semantic_multi_positive",
        load_queries(args.repaired_release, ["semantic_development.jsonl"]),
        all_items,
    )
    d1_localized = make_view(
        "D1_localized_evaluation",
        load_queries(args.repaired_release, ["localized_development.jsonl"]),
        all_items,
    )
    d1_direction = make_view(
        "D1_direction_sensitive",
        load_queries(args.repaired_release, ["direction_development.jsonl"]),
        all_items,
    )
    # No verified Forest/RSCC/localized/long-series rows exist in the frozen
    # release, so D2 and D3 keep the exact query cohort unchanged explicitly.
    d2_exact = make_view("D2_verified_forest_rscc", d1_exact["rows"], all_items)
    d3_exact = make_view("D3_verified_localized_long_series", d2_exact["rows"], all_items)
    views = [d0_exact, d1_exact, d1_stable, d1_semantic, d1_localized, d1_direction, d2_exact, d3_exact]
    for view in views:
        view["items"] = all_items

    gallery_ids = sorted({item_id for view in views for item_id in view["gallery_ids"]})
    gallery_rows = pair_rows(gallery_ids, all_items)
    model, text_encoder, encoder, image_batch_fn, load_text_batch_fn = load_frozen_runtime(args)
    device = torch.device("cuda", torch.cuda.current_device())
    started = time.monotonic()
    pair_embeddings = extract_pair_embeddings(
        gallery_rows,
        model,
        encoder,
        image_batch_fn,
        device,
        args.image_size,
        args.extract_batch_size,
    )
    cache_path = args.output_dir / "frozen_pair_embeddings.pt"
    torch.save({"schema_version": "qcpr-frozen-b1-pair-embedding-cache-v1", "item_ids": gallery_ids, "embeddings": pair_embeddings}, cache_path)
    detailed_all = {}
    for view in views:
        detailed_all[view["name"]] = evaluate_view(
            view,
            pair_embeddings,
            model,
            text_encoder,
            load_text_batch_fn,
            device,
            args.text_batch,
            args.bootstrap_rounds,
        )
    results = {}
    for name, value in detailed_all.items():
        results[name] = {key: item for key, item in value.items() if key != "per_query"}
    detailed_views = {}
    for view in [d0_exact, d1_exact, d2_exact, d3_exact]:
        detailed_views[view["name"]] = detailed_all[view["name"]]
    comparisons = {
        "D0_vs_D1_exact": paired_bootstrap(detailed_views["D0_current_exact_core"], detailed_views["D1_repaired_exact"], args.bootstrap_rounds, 20260805),
        "D1_vs_D2_exact": paired_bootstrap(detailed_views["D1_repaired_exact"], detailed_views["D2_verified_forest_rscc"], args.bootstrap_rounds, 20260815),
        "D2_vs_D3_exact": paired_bootstrap(detailed_views["D2_verified_forest_rscc"], detailed_views["D3_verified_localized_long_series"], args.bootstrap_rounds, 20260825),
    }
    state = {
        "schema_version": "qcpr-retrieval-data-only-comparison-v2",
        "status": "PASS_FROZEN_B1_DATA_ONLY_METRICS",
        "same_frozen_model": True,
        "training_run": False,
        "mask_access": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "gallery_item_count": len(gallery_ids),
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "views": results,
        "paired_bootstrap": comparisons,
        "verified_additions": {
            "forest_query_count": 0,
            "rscc_query_count": 0,
            "localized_query_count": 0,
            "long_series_query_count": 0,
            "d2_equals_d1_exact_cohort": True,
            "d3_equals_d2_exact_cohort": True,
        },
        "improvement_claim": False,
        "elapsed_seconds": time.monotonic() - started,
        "notes": [
            "D0/D1/D2/D3 exact views use their frozen development query cohorts and positive-item galleries.",
            "D2/D3 add zero verified rows because Forest, RSCC, localized and long-series text gates remain closed.",
            "Semantic, stable, localized and direction results are evaluation metrics, not training authorization.",
        ],
    }
    write_json(args.output_dir / "comparison.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
