from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a pair-retrieval evaluation report into a compact JSON artifact.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _top_transitions(summary: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    counts = dict(summary.get("dominant_transition_counts", {}))
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    return [{"transition": key, "count": value} for key, value in ordered[:limit]]


def _compact_metrics(metrics: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: metrics[key] for key in keys if key in metrics}


def build_summary(payload: dict[str, Any], eval_json: Path) -> dict[str, Any]:
    overall = dict(payload.get("overall", payload))
    by_source = dict(payload.get("by_source", {}))
    dataset_summary = dict(payload.get("dataset_summary", {}))
    transition_summary = dict(payload.get("transition_summary", {}))

    source_highlights = {}
    for source, metrics in by_source.items():
        source_highlights[source] = _compact_metrics(
            metrics,
            [
                "recall@5",
                "mAP",
                "pair_to_text_recall@5",
                "transition_recall@5",
                "transition_top1_hit_rate",
            ],
        )

    summary = {
        "eval_json": str(eval_json),
        "overall_highlights": _compact_metrics(
            overall,
            [
                "recall@1",
                "recall@5",
                "recall@10",
                "mAP",
                "MRR",
                "median_rank",
                "pair_to_text_recall@1",
                "pair_to_text_recall@5",
                "pair_to_text_recall@10",
                "pair_to_text_MRR",
                "reversed_pair_sanity_accuracy",
                "transition_recall@1",
                "transition_recall@5",
                "transition_recall@10",
                "transition_MRR",
                "transition_top1_hit_rate",
                "mean_transition_similarity_top5",
                "pair_sample_fraction",
            ],
        ),
        "dataset_summary": {
            "num_samples": dataset_summary.get("num_samples"),
            "num_unique_pairs": dataset_summary.get("num_unique_pairs"),
            "num_caption_rows": dataset_summary.get("num_caption_rows"),
            "num_pair_samples": dataset_summary.get("num_pair_samples"),
            "num_caption_samples": dataset_summary.get("num_caption_samples"),
            "source_counts": dataset_summary.get("source_counts", {}),
            "split_pair_counts": dataset_summary.get("split_pair_counts", {}),
            "top_dominant_transitions": _top_transitions(dataset_summary),
        },
        "source_highlights": source_highlights,
        "transition_summary": {
            "pair_only_top_transitions": _top_transitions(transition_summary.get("pair_only", {})),
            "caption_or_grounded_text_top_transitions": _top_transitions(transition_summary.get("caption_or_grounded_text", {})),
        },
    }
    return summary


def main() -> int:
    args = parse_args()
    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    summary = build_summary(payload, args.eval_json)
    rendered = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
