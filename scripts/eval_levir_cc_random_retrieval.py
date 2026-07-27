from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic random LEVIR-CC retrieval baseline.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _average_precision(relevances: list[bool]) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / hits if hits else 0.0


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.manifest)
    pair_ids = []
    seen = set()
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in seen:
            seen.add(pair_id)
            pair_ids.append(pair_id)
    captions_by_pair: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        captions_by_pair[str(row["pair_id"])].append(str(row["sample_id"]))

    rng = random.Random(args.seed)
    text_ranks = []
    text_ap = []
    text_hits = {1: 0, 5: 0, 10: 0}
    for row in rows:
        ranking = list(pair_ids)
        rng.shuffle(ranking)
        relevances = [candidate == str(row["pair_id"]) for candidate in ranking]
        for k in text_hits:
            text_hits[k] += 1 if any(relevances[:k]) else 0
        text_ranks.append(next((idx for idx, rel in enumerate(relevances, start=1) if rel), len(ranking) + 1))
        text_ap.append(_average_precision(relevances))

    pair_ranks = []
    pair_hits = {1: 0, 5: 0, 10: 0}
    for pair_id in pair_ids:
        ranking = [row["sample_id"] for row in rows]
        rng.shuffle(ranking)
        positives = set(captions_by_pair[pair_id])
        relevances = [candidate in positives for candidate in ranking]
        for k in pair_hits:
            pair_hits[k] += 1 if any(relevances[:k]) else 0
        pair_ranks.append(next((idx for idx, rel in enumerate(relevances, start=1) if rel), len(ranking) + 1))

    report = {
        "baseline": "random_retrieval",
        "seed": args.seed,
        "num_unique_pairs": len(pair_ids),
        "num_caption_queries": len(rows),
        "recall@1": text_hits[1] / max(len(rows), 1),
        "recall@5": text_hits[5] / max(len(rows), 1),
        "recall@10": text_hits[10] / max(len(rows), 1),
        "MRR": sum(1.0 / rank for rank in text_ranks) / max(len(text_ranks), 1),
        "median_rank": float(np.median(text_ranks)) if text_ranks else 0.0,
        "mAP": sum(text_ap) / max(len(text_ap), 1),
        "pair_to_text_recall@1": pair_hits[1] / max(len(pair_ids), 1),
        "pair_to_text_recall@5": pair_hits[5] / max(len(pair_ids), 1),
        "pair_to_text_recall@10": pair_hits[10] / max(len(pair_ids), 1),
        "pair_to_text_MRR": sum(1.0 / rank for rank in pair_ranks) / max(len(pair_ranks), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
