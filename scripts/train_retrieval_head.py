from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.training.configs import TrainingConfig
from land_change_detection.training.losses import info_nce_loss, triplet_ranking_loss
from land_change_detection.training.retrieval_datasets import ManifestRetrievalDataset, load_manifest_records
from land_change_detection.training.samplers import build_triplets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny retrieval-head training scaffold.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics.")
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ManifestRetrievalDataset(load_manifest_records(args.manifest))
    config = TrainingConfig(epochs=args.epochs)
    triplets = build_triplets(dataset)

    triplet_losses: list[float] = []
    contrastive_losses: list[float] = []
    for triplet in triplets:
        triplet_losses.append(
            triplet_ranking_loss(triplet.anchor.features, triplet.positive.features, triplet.negative.features, margin=config.margin)
        )
        contrastive_losses.append(
            info_nce_loss(
                triplet.anchor.features,
                triplet.positive.features,
                negatives=[triplet.negative.features],
                temperature=config.temperature,
            )
        )

    metrics = {
        "epochs": config.epochs,
        "samples": len(dataset),
        "triplets": len(triplets),
        "mean_triplet_loss": sum(triplet_losses) / len(triplet_losses) if triplet_losses else 0.0,
        "mean_info_nce_loss": sum(contrastive_losses) / len(contrastive_losses) if contrastive_losses else 0.0,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "checkpoint.json").write_text(json.dumps({"status": "fake_checkpoint", "samples": len(dataset)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
