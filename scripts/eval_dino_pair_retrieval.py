from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dino_pair_retrieval import (
    build_dataloader,
    choose_device,
    load_retrieval_samples,
    run_epoch,
    RetrievalSample,
)
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINO/simple-patch pair retrieval.")
    parser.add_argument("--levir-manifest", type=Path, default=None)
    parser.add_argument("--pair-manifest", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-backbone", choices=("simple_patch", "dinov2"), default="simple_patch")
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pair-loss", choices=("supervised", "soft"), default="supervised")
    parser.add_argument("--lambda-pair", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def _summarize_samples(samples: list[RetrievalSample]) -> dict[str, object]:
    source_counts = Counter(sample.source for sample in samples)
    transition_counts = Counter(sample.dominant_transition for sample in samples if sample.dominant_transition)
    pair_samples = [sample for sample in samples if sample.transition_histogram]
    caption_samples = [sample for sample in samples if sample.caption]
    return {
        "num_samples": len(samples),
        "num_pair_samples": len(pair_samples),
        "num_caption_samples": len(caption_samples),
        "source_counts": dict(source_counts),
        "dominant_transition_counts": dict(transition_counts),
    }


def _metrics_for_subset(
    samples: list[RetrievalSample],
    model: DINOChangeRetriever,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if not samples:
        return {}
    subset_args = argparse.Namespace(**vars(args))
    subset_args.max_train_samples = None
    loader = build_dataloader(samples, subset_args, shuffle=False)
    return run_epoch(model, loader, None, subset_args, device)


def main() -> int:
    args = parse_args()
    samples = load_retrieval_samples(args.levir_manifest, list(args.pair_manifest))
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone=args.visual_backbone,
            dinov2_model_path=str(args.dinov2_model_path) if args.dinov2_model_path else None,
            local_files_only=args.local_files_only,
            image_size=args.image_size,
        )
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = _metrics_for_subset(samples, model, args, device)
    by_source = {
        source: _metrics_for_subset([sample for sample in samples if sample.source == source], model, args, device)
        for source in sorted({sample.source for sample in samples})
    }
    pair_only_samples = [sample for sample in samples if sample.transition_histogram]
    caption_only_samples = [sample for sample in samples if sample.caption]
    report = dict(metrics)
    report["overall"] = metrics
    report["by_source"] = by_source
    report["dataset_summary"] = _summarize_samples(samples)
    report["transition_summary"] = {
        "pair_only": _summarize_samples(pair_only_samples),
        "caption_or_grounded_text": _summarize_samples(caption_only_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
