#!/usr/bin/env python3
"""Analyze existing direct TemporalSigLIP ranking artifacts without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from qcpr_siglip2.data.manifest import group_rows_by_pair, load_exact_pair_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RANKINGS_PT",
        help="repeat for stage_a, stage_b_step1024 and stage_b_step2048",
    )
    return parser.parse_args()


def rank_metrics(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        raise ValueError("empty rank list")
    return {
        "query_count": len(ranks),
        "mrr_full": mean(1.0 / rank for rank in ranks),
        "hit_at_1": mean(rank <= 1 for rank in ranks),
        "hit_at_5": mean(rank <= 5 for rank in ranks),
        "hit_at_10": mean(rank <= 10 for rank in ranks),
        "hit_at_50": mean(rank <= 50 for rank in ranks),
        "hit_at_100": mean(rank <= 100 for rank in ranks),
        "mean_rank": mean(ranks),
        "median_rank": median(ranks),
    }


def ranks_for_rows(
    rankings: torch.Tensor,
    rows: list[dict[str, Any]],
    pair_ids: list[str],
) -> list[int]:
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    output: list[int] = []
    for query_index, row in enumerate(rows):
        positives = [
            pair_index[str(item)]
            for item in row.get("positive_pair_ids", [])
            if str(item) in pair_index
        ]
        if not positives:
            raise ValueError(f"query has no gallery positive: {row['caption_id']}")
        ranking = rankings[query_index].tolist()
        positions = [ranking.index(index) + 1 for index in positives]
        output.append(min(positions))
    return output


def main() -> int:
    args = parse_args()
    rows = load_exact_pair_rows(args.development_manifest, split="development")
    groups = group_rows_by_pair(rows)
    pair_ids = [str(pair_id) for pair_id in groups]
    arms: dict[str, str] = {}
    for value in args.arm:
        if "=" not in value:
            raise ValueError(f"invalid --arm value: {value}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid --arm value: {value}")
        arms[name] = path

    report: dict[str, Any] = {
        "protocol": "QCPR_EXACT_FULL_GALLERY_DIRECT_TEMPORALSIGLIP",
        "development_manifest": str(args.development_manifest),
        "development_manifest_sha256": sha256(Path(args.development_manifest)),
        "gallery_size": len(pair_ids),
        "query_count": len(rows),
        "pair_id_order": "group_rows_by_pair(first-seen order)",
        "arms": {},
        "limitations": {
            "generic_no_change_primary_metric": "NOT_APPLICABLE_exact_manifest_excludes_it",
            "source_classification_separability": "NOT_RUN_rankings_do_not_contain_embeddings",
            "metric_labels": "hit_at_k_is_hit_rate; mrr_full_is_full_gallery_MRR",
        },
    }
    for name, path_value in arms.items():
        path = Path(path_value)
        rankings = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(rankings, torch.Tensor) or rankings.ndim != 2:
            raise ValueError(f"invalid rankings tensor: {path}")
        if tuple(rankings.shape) != (len(rows), len(pair_ids)):
            raise ValueError(
                f"ranking shape mismatch for {name}: {tuple(rankings.shape)} "
                f"!= {(len(rows), len(pair_ids))}"
            )
        all_ranks = ranks_for_rows(rankings, rows, pair_ids)
        by_source: dict[str, dict[str, Any]] = {}
        for source in sorted({str(row["dataset_name"]) for row in rows}):
            selected_rows = [
                (index, row)
                for index, row in enumerate(rows)
                if str(row["dataset_name"]) == source
            ]
            selected_ranks = [all_ranks[index] for index, _ in selected_rows]
            by_source[source] = rank_metrics(selected_ranks)
        by_scope: dict[str, dict[str, Any]] = {}
        for scope in sorted({str(row.get("query_scope")) for row in rows}):
            selected = [
                all_ranks[index]
                for index, row in enumerate(rows)
                if str(row.get("query_scope")) == scope
            ]
            by_scope[scope] = rank_metrics(selected)
        report["arms"][name] = {
            "ranking_path": str(path),
            "ranking_sha256": sha256(path),
            "overall": rank_metrics(all_ranks),
            "by_source": by_source,
            "by_query_scope": by_scope,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
