from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.training.metrics import mean_average_precision, recall_at_k, transition_consistency_score
from land_change_detection.training.retrieval_datasets import ManifestRetrievalDataset, load_manifest_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toy retrieval manifest metrics.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ManifestRetrievalDataset(load_manifest_records(args.manifest))
    relevance_lists: list[list[bool]] = []
    transition_scores: list[float] = []
    for record in dataset.records:
        ranked_ids = list(record.positives) + list(record.negatives)
        relevance_lists.append([candidate in record.positives for candidate in ranked_ids])
        transition_scores.append(transition_consistency_score(record.transition_label, [record.transition_label] if record.transition_label else []))

    metrics = {
        "samples": len(dataset),
        "Recall@1": sum(recall_at_k(values, 1) for values in relevance_lists) / len(relevance_lists) if relevance_lists else 0.0,
        "Recall@5": sum(recall_at_k(values, 5) for values in relevance_lists) / len(relevance_lists) if relevance_lists else 0.0,
        "Recall@10": sum(recall_at_k(values, 10) for values in relevance_lists) / len(relevance_lists) if relevance_lists else 0.0,
        "mAP": mean_average_precision(relevance_lists),
        "transition_consistency": sum(transition_scores) / len(transition_scores) if transition_scores else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
