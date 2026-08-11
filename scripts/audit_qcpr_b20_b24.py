#!/usr/bin/env python3
"""Audit B20/B24 checkpoints on the immutable r19g development gallery.

This script consumes already-produced full-ranking tensors.  It never opens
images, masks, or training data, and it does not retrain or alter historical
run directories.  The paired bootstrap unit is the physical pair: all query
rows attached to one physical pair are retained together in each resample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_name(pair_id: str) -> str:
    return str(pair_id).split(":", 1)[0]


def load_rankings(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"ranking artifact is not a dictionary: {path}")
    required = {"global_scores", "positive_mask", "ignored_mask"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ranking artifact missing {sorted(missing)}: {path}")
    scores = payload["global_scores"].float()
    positive = payload["positive_mask"].bool()
    ignored = payload["ignored_mask"].bool()
    if scores.shape != positive.shape or scores.shape != ignored.shape:
        raise ValueError(f"ranking tensors have inconsistent shapes: {path}")
    if not torch.isfinite(scores).all():
        raise ValueError(f"non-finite scores: {path}")
    if torch.any(positive & ignored):
        raise ValueError(f"positive/ignored overlap: {path}")
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError(f"query without a positive: {path}")
    return {"scores": scores, "positive": positive, "ignored": ignored}


def first_positive_ranks(scores: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=1, descending=True)
    ranked = positive.gather(1, order)
    first = ranked.float().argmax(dim=1) + 1
    missing = ranked.sum(dim=1) == 0
    return torch.where(missing, torch.full_like(first, scores.shape[1] + 1), first)


def rank_metrics(ranks: torch.Tensor) -> dict[str, float]:
    values = ranks.float()
    return {
        "mrr_full": float((1.0 / values).mean()),
        "candidate_hit_at_1": float((ranks <= 1).float().mean()),
        "candidate_hit_at_5": float((ranks <= 5).float().mean()),
        "candidate_hit_at_10": float((ranks <= 10).float().mean()),
        "candidate_hit_at_50": float((ranks <= 50).float().mean()),
        "candidate_hit_at_100": float((ranks <= 100).float().mean()),
        "mean_rank": float(values.mean()),
        "median_rank": float(values.median()),
    }


def within_source_ranks(
    scores: torch.Tensor,
    positive: torch.Tensor,
    pair_ids: list[str],
    query_sources: list[str],
) -> torch.Tensor:
    result: list[int] = []
    pair_sources = [source_name(pair_id) for pair_id in pair_ids]
    for row_index, source in enumerate(query_sources):
        columns = [index for index, item_source in enumerate(pair_sources) if item_source == source]
        if not columns:
            raise ValueError(f"no within-source gallery for {source}")
        local_scores = scores[row_index, columns].view(1, -1)
        local_positive = positive[row_index, columns].view(1, -1)
        result.append(int(first_positive_ranks(local_scores, local_positive)[0]))
    return torch.tensor(result, dtype=torch.long)


def bootstrap_delta(
    b24: torch.Tensor,
    b20: torch.Tensor,
    cluster_ids: list[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if b24.shape != b20.shape or b24.ndim != 1:
        raise ValueError("bootstrap rank vectors must have the same one-dimensional shape")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        grouped[str(cluster)].append(index)
    group_indices = [torch.tensor(indices, dtype=torch.long) for indices in grouped.values()]
    generator = torch.Generator().manual_seed(seed)
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = torch.randint(len(group_indices), (len(group_indices),), generator=generator)
        query_indices = torch.cat([group_indices[index] for index in sampled.tolist()])
        left = rank_metrics(b24[query_indices])
        right = rank_metrics(b20[query_indices])
        for name in left:
            deltas[name].append(left[name] - right[name])

    result: dict[str, Any] = {
        "unit": "physical_pair",
        "physical_pair_count": len(group_indices),
        "query_count": int(b24.numel()),
        "replicates": replicates,
        "seed": seed,
        "delta_definition": "B24 - B20",
        "metrics": {},
    }
    for name, values in deltas.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        lower, median, upper = torch.quantile(
            tensor,
            torch.tensor([0.025, 0.5, 0.975], dtype=tensor.dtype),
        )
        nonpositive = float((tensor <= 0).float().mean())
        nonnegative = float((tensor >= 0).float().mean())
        result["metrics"][name] = {
            "point_estimate": float(rank_metrics(b24)[name] - rank_metrics(b20)[name]),
            "ci95": {"lower": float(lower), "median": float(median), "upper": float(upper)},
            "sign_consistency": float((tensor > 0).float().mean()),
            "empirical_two_sided_sign_p": min(1.0, 2.0 * min(nonpositive, nonnegative)),
        }
    return result


def parse_top100(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def top20_records(
    *,
    scores: torch.Tensor,
    positive: torch.Tensor,
    query_rows: list[dict[str, Any]],
    pair_ids: list[str],
    checkpoint_id: str,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    order = scores.argsort(dim=1, descending=True)
    output: list[dict[str, Any]] = []
    for query_index, row in enumerate(query_rows):
        ranked = order[query_index]
        positive_indices = torch.where(positive[query_index])[0].tolist()
        exact_rank = min(
            int((ranked == positive_index).nonzero(as_tuple=False)[0, 0]) + 1
            for positive_index in positive_indices
        )
        exact_index = positive_indices[0]
        exact_score = float(scores[query_index, exact_index])
        top_indices = ranked[:20].tolist()
        top_scores = [float(scores[query_index, index]) for index in top_indices]
        top_items = [
            {
                "item_id": pair_ids[index],
                "rank": position + 1,
                "score": top_scores[position],
                "candidate_source": source_name(pair_ids[index]),
            }
            for position, index in enumerate(top_indices)
        ]
        output.append(
            {
                "query_id": str(row["query_id"]),
                "query_text": str(row.get("text", "")),
                "query_source": str(row.get("provenance", {}).get("source_dataset", source_name(str(row["source_pair_id"])))),
                "exact_item_id": str(row["source_pair_id"]),
                "exact_rank": exact_rank,
                "exact_score": exact_score,
                "score_margin_top1_top2": (
                    top_scores[0] - top_scores[1] if len(top_scores) >= 2 else None
                ),
                "score_margin_exact_vs_top1": exact_score - top_scores[0],
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha": checkpoint_sha256,
                "top20": top_items,
            }
        )
    return output


def crosscheck_top100(
    records: list[dict[str, Any]],
    serialized: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mismatches = []
    for record in records:
        row = serialized.get(record["query_id"])
        if row is None:
            mismatches.append({"query_id": record["query_id"], "reason": "missing_query"})
            continue
        expected = [item["item_id"] for item in record["top20"]]
        observed = [str(item) for item in row.get("top_pair_ids", [])[:20]]
        if expected != observed or int(row.get("rank", -1)) != int(record["exact_rank"]):
            mismatches.append(
                {
                    "query_id": record["query_id"],
                    "expected_rank": record["exact_rank"],
                    "observed_rank": row.get("rank"),
                    "top20_equal": expected == observed,
                }
            )
    return {"rows": len(records), "mismatches": len(mismatches), "examples": mismatches[:10]}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b20-rankings", type=Path, required=True)
    parser.add_argument("--b20-top100", type=Path, required=True)
    parser.add_argument("--b20-evaluation", type=Path, required=True)
    parser.add_argument("--b20-checkpoint-sha", required=True)
    parser.add_argument("--b20-checkpoint-id", default="B20_step_1140")
    parser.add_argument("--b20-run-release-sha", required=True)
    parser.add_argument("--b24-rankings", type=Path, required=True)
    parser.add_argument("--b24-top100", type=Path, required=True)
    parser.add_argument("--b24-evaluation", type=Path, required=True)
    parser.add_argument("--b24-checkpoint-sha", required=True)
    parser.add_argument("--b24-checkpoint-id", default="B24_step_1368")
    parser.add_argument("--b24-run-release-sha", required=True)
    parser.add_argument("--expected-dev-manifest-sha", required=True)
    parser.add_argument("--expected-gallery-sha", required=True)
    parser.add_argument("--expected-query-sha", required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    query_rows = read_jsonl(args.development_manifest)
    query_ids = [str(row["query_id"]) for row in query_rows]
    pair_ids: list[str] = []
    for row in query_rows:
        pair_id = str(row["source_pair_id"])
        if pair_id not in pair_ids:
            pair_ids.append(pair_id)
    query_sources = [
        str(row.get("provenance", {}).get("source_dataset", source_name(row["source_pair_id"])))
        for row in query_rows
    ]
    pair_sha = sequence_sha256(pair_ids)
    query_sha = sequence_sha256(query_ids)
    manifest_sha = sha256_file(args.development_manifest)
    if manifest_sha != args.expected_dev_manifest_sha:
        raise ValueError(f"development manifest SHA mismatch: {manifest_sha}")
    if pair_sha != args.expected_gallery_sha:
        raise ValueError(f"gallery SHA mismatch: {pair_sha}")
    if query_sha != args.expected_query_sha:
        raise ValueError(f"query SHA mismatch: {query_sha}")

    arms = {
        "B20": {
            "rankings": load_rankings(args.b20_rankings),
            "top100": parse_top100(args.b20_top100),
            "evaluation": json.loads(args.b20_evaluation.read_text()),
            "checkpoint_sha256": args.b20_checkpoint_sha,
            "checkpoint_id": args.b20_checkpoint_id,
            "run_release_sha": args.b20_run_release_sha,
        },
        "B24": {
            "rankings": load_rankings(args.b24_rankings),
            "top100": parse_top100(args.b24_top100),
            "evaluation": json.loads(args.b24_evaluation.read_text()),
            "checkpoint_sha256": args.b24_checkpoint_sha,
            "checkpoint_id": args.b24_checkpoint_id,
            "run_release_sha": args.b24_run_release_sha,
        },
    }
    for arm in arms.values():
        tensors = arm["rankings"]
        if tuple(tensors["scores"].shape) != (len(query_rows), len(pair_ids)):
            raise ValueError("ranking shape does not match current r19g manifest")
        arm["ranks"] = first_positive_ranks(tensors["scores"], tensors["positive"])
        arm["within_ranks"] = within_source_ranks(
            tensors["scores"], tensors["positive"], pair_ids, query_sources
        )
        arm["metrics"] = rank_metrics(arm["ranks"])
        arm["within_metrics"] = rank_metrics(arm["within_ranks"])
        arm["top20_records"] = top20_records(
            scores=tensors["scores"],
            positive=tensors["positive"],
            query_rows=query_rows,
            pair_ids=pair_ids,
            checkpoint_id=arm["checkpoint_id"],
            checkpoint_sha256=arm["checkpoint_sha256"],
        )
        arm["top100_crosscheck"] = crosscheck_top100(
            arm["top20_records"], arm["top100"]
        )
        arm["full_rankings_sha256"] = sha256_file(
            args.b20_rankings if arm["checkpoint_id"].startswith("B20") else args.b24_rankings
        )

    cluster_ids = [str(row["source_pair_id"]) for row in query_rows]
    bootstrap = {
        "overall": bootstrap_delta(
            arms["B24"]["ranks"], arms["B20"]["ranks"], cluster_ids,
            seed=args.bootstrap_seed, replicates=args.bootstrap_replicates,
        ),
        "within_source": bootstrap_delta(
            arms["B24"]["within_ranks"], arms["B20"]["within_ranks"], cluster_ids,
            seed=args.bootstrap_seed + 1, replicates=args.bootstrap_replicates,
        ),
    }

    per_source: dict[str, Any] = {}
    for source in sorted(set(query_sources)):
        indices = [index for index, item in enumerate(query_sources) if item == source]
        per_source[source] = {
            "query_count": len(indices),
            "B20": rank_metrics(arms["B20"]["ranks"][indices]),
            "B24": rank_metrics(arms["B24"]["ranks"][indices]),
            "delta_B24_minus_B20": {
                name: rank_metrics(arms["B24"]["ranks"][indices])[name]
                - rank_metrics(arms["B20"]["ranks"][indices])[name]
                for name in arms["B20"]["metrics"]
            },
        }

    input_audit = {
        "authoritative_dataset_release_sha256": args.release_sha,
        "development_manifest": str(args.development_manifest),
        "development_manifest_sha256": manifest_sha,
        "ordered_gallery_ids_sha256": pair_sha,
        "ordered_query_ids_sha256": query_sha,
        "query_count": len(query_rows),
        "gallery_count": len(pair_ids),
        "query_ids_unique": len(set(query_ids)) == len(query_ids),
        "gallery_ids_unique": len(set(pair_ids)) == len(pair_ids),
        "source_counts": {
            source: query_sources.count(source) for source in sorted(set(query_sources))
        },
        "run_release_field_comparison": {
            "B20_recorded_release_sha256": args.b20_run_release_sha,
            "B24_recorded_release_sha256": args.b24_run_release_sha,
            "authoritative_release_sha256": args.release_sha,
            "match": args.b20_run_release_sha == args.release_sha
            and args.b24_run_release_sha == args.release_sha,
            "interpretation": "manifest_level_match_but_historical_run_release_field_stale"
            if args.b20_run_release_sha != args.release_sha
            or args.b24_run_release_sha != args.release_sha
            else "full_release_field_match",
        },
    }
    arms_summary = {
        name: {
            "checkpoint_id": arm["checkpoint_id"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "run_release_sha256": arm["run_release_sha"],
            "full_rankings_path": str(
                args.b20_rankings if name == "B20" else args.b24_rankings
            ),
            "full_rankings_sha256": arm["full_rankings_sha256"],
            "metrics": arm["metrics"],
            "within_source_metrics": arm["within_metrics"],
            "top100_crosscheck": arm["top100_crosscheck"],
        }
        for name, arm in arms.items()
    }
    summary = {
        "status": "PASS_MANIFEST_MATCH_STALE_RUN_RELEASE_FIELD"
        if not input_audit["run_release_field_comparison"]["match"]
        else "PASS",
        "producer_sha": args.producer_sha,
        "protocol": "QCPR_EXACT_FULL_GALLERY",
        "input_audit": input_audit,
        "arms": arms_summary,
        "bootstrap": bootstrap,
        "per_source": per_source,
        "route_gap_note": "ROUTE_GAP_OBSERVED is reported separately; it is not an architecture-defect claim.",
        "semantic_status": "NOT_EVALUATED; current r19g semantic_eval_ready=false",
    }
    write_json(args.output_dir / "qcpr_b20_vs_b24_bootstrap.json", summary)
    write_json(args.output_dir / "qcpr_dev_top20_summary.json", {
        "status": summary["status"],
        "producer_sha": args.producer_sha,
        "release_sha256": args.release_sha,
        "development_manifest_sha256": manifest_sha,
        "ordered_gallery_ids_sha256": pair_sha,
        "ordered_query_ids_sha256": query_sha,
        "query_count": len(query_rows),
        "gallery_count": len(pair_ids),
        "arms": arms_summary,
        "top20_crosscheck": {
            name: arm["top100_crosscheck"] for name, arm in arms.items()
        },
    })
    with (args.output_dir / "qcpr_dev_top20_for_dataset_agent.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(query_rows)):
            record = {
                "query_id": query_ids[index],
                "query_text": str(query_rows[index].get("text", "")),
                "query_source": query_sources[index],
                "exact_item_id": str(query_rows[index]["source_pair_id"]),
                "B20": arms["B20"]["top20_records"][index],
                "B24": arms["B24"]["top20_records"][index],
                "authoritative_release_sha256": args.release_sha,
                "development_manifest_sha256": manifest_sha,
                "ordered_gallery_ids_sha256": pair_sha,
                "ordered_query_ids_sha256": query_sha,
                "producer_sha": args.producer_sha,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    md = [
        "# B20 vs B24 and Top-20 Audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "The paired bootstrap unit is the physical pair; all query rows for a sampled pair are retained together.",
        "The per-query export contains both B20 (step 1140) and B24 (step 1368) on the same r19g gallery.",
        "",
        "## Inputs",
        "",
        f"- authoritative release SHA: `{args.release_sha}`",
        f"- development manifest SHA: `{manifest_sha}`",
        f"- gallery IDs SHA: `{pair_sha}`",
        f"- query IDs SHA: `{query_sha}`",
        f"- gallery/query: `{len(pair_ids)}` / `{len(query_rows)}`",
        f"- historical run release field match: `{input_audit['run_release_field_comparison']['match']}`",
        "",
        "## Point metrics",
        "",
        "| Arm | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@50 | Hit@100 | mean rank | median rank | within MRR | within Hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("B20", "B24"):
        m = arms[name]["metrics"]
        w = arms[name]["within_metrics"]
        md.append(
            f"| {name} | {m['mrr_full']:.8f} | {m['candidate_hit_at_1']:.8f} | {m['candidate_hit_at_5']:.8f} | {m['candidate_hit_at_10']:.8f} | {m['candidate_hit_at_50']:.8f} | {m['candidate_hit_at_100']:.8f} | {m['mean_rank']:.4f} | {m['median_rank']:.1f} | {w['mrr_full']:.8f} | {w['candidate_hit_at_10']:.8f} |"
        )
    md.extend([
        "",
        "## Bootstrap",
        "",
        f"- replicates: `{args.bootstrap_replicates}`; seed: `{args.bootstrap_seed}`; delta: `B24 - B20`.",
        "- See the JSON for percentile 95% intervals and empirical two-sided sign p-values.",
        "",
        "## Integrity",
        "",
        f"- B20 Top-100 cross-check: `{arms['B20']['top100_crosscheck']}`",
        f"- B24 Top-100 cross-check: `{arms['B24']['top100_crosscheck']}`",
        "- Semantic multi-positive evaluation is not performed because the current dataset handoff reports semantic evaluation not ready.",
        "- Historical run metadata records a stale release-root hash, but the exact development manifest, ordered query IDs and ordered gallery IDs match the authoritative r19g inputs. This is retained as a provenance warning, not silently hidden.",
    ])
    (args.output_dir / "qcpr_b20_vs_b24_bootstrap.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.output_dir / "qcpr_dev_top20_summary.md").write_text(
        "# Top-20 Forensic Export\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Rows: `{len(query_rows)}`; gallery: `{len(pair_ids)}`. Both B20 and B24 are included in `qcpr_dev_top20_for_dataset_agent.jsonl`.\n\n"
        f"Release SHA: `{args.release_sha}`\n\n"
        f"Top-100 cross-checks: B20 `{arms['B20']['top100_crosscheck']}`, B24 `{arms['B24']['top100_crosscheck']}`.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
