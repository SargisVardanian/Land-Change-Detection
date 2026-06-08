from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the semantic-first baseline on semantic manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-channels", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--run-tag", default="semantic_baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from land_change_detection.models import SemanticChangeModelConfig, build_semantic_change_model
    from land_change_detection.training.semantic_manifest_dataset import SemanticManifestDataset, load_semantic_manifest

    records = load_semantic_manifest(args.manifest)
    dataset = SemanticManifestDataset(records, input_channels=args.input_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_semantic_change_model(
        SemanticChangeModelConfig(input_channels=args.input_channels, num_classes=args.num_classes, base_channels=8)
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    change_acc_values: list[float] = []
    semantic_supervised_pixels = 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch["before"], batch["after"])
            pred_change = output.binary_change_map
            target_change = (batch["change_target"] > 0.5).to(torch.uint8)
            change_acc_values.append(float((pred_change == target_change).float().mean().cpu()))

            semantic_supervised_pixels += int(((batch["before_target"] >= 0) & (batch["after_target"] >= 0)).sum().cpu())

    metrics = {
        "samples": len(dataset),
        "change_accuracy": sum(change_acc_values) / len(change_acc_values) if change_acc_values else 0.0,
        "semantic_supervised_pixels": semantic_supervised_pixels,
        "input_channels": args.input_channels,
        "num_classes": args.num_classes,
        "run_tag": args.run_tag,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
