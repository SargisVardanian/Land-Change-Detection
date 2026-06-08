from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the semantic-first baseline model from semantic manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-channels", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--run-tag", default="semantic_baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from land_change_detection.models import SemanticChangeModelConfig, build_semantic_change_model
    from land_change_detection.training.losses import semantic_change_loss
    from land_change_detection.training.semantic_manifest_dataset import SemanticManifestDataset, load_semantic_manifest

    records = load_semantic_manifest(args.manifest)
    dataset = SemanticManifestDataset(records, input_channels=args.input_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_semantic_change_model(
        SemanticChangeModelConfig(input_channels=args.input_channels, num_classes=args.num_classes, base_channels=8)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_losses: list[float] = []
    for _ in range(args.epochs):
        batch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["before"], batch["after"])
            loss = semantic_change_loss(
                output,
                batch["before_target"],
                batch["after_target"],
                change_target=batch["change_target"],
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses) if batch_losses else 0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "semantic_change_model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "samples": len(dataset),
                "mean_epoch_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
                "epoch_losses": epoch_losses,
                "input_channels": args.input_channels,
                "num_classes": args.num_classes,
                "run_tag": args.run_tag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
