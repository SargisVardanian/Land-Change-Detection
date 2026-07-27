from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone SemanticChangeBaseline for T1/T2 land-cover transitions.")
    parser.add_argument("--manifest", default=None, help="Backward-compatible smoke mode: use one manifest for train and val.")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--input-channels", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--run-tag", default="semantic_change_baseline")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _json_ready(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_ready(value) for value in payload]
    if hasattr(payload, "item"):
        return payload.item()
    return payload


def _load_supervised_dataset(path: str, input_channels: int):
    from land_change_detection.training.semantic_manifest_dataset import (
        SemanticManifestDataset,
        filter_semantic_supervised_records,
        load_semantic_manifest,
    )

    records = filter_semantic_supervised_records(load_semantic_manifest(path))
    if not records:
        raise RuntimeError(
            f"No semantic-transition supervised rows found in {path}. "
            "Rows must have task=semantic_transition_segmentation plus semantic_before_path and semantic_after_path."
        )
    return SemanticManifestDataset(records, input_channels=input_channels)


def _mean_loss(losses: list[float]) -> float:
    return sum(losses) / len(losses) if losses else 0.0


def _aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    keys = sorted({key for item in items for key, value in item.items() if isinstance(value, int | float) and value is not None})
    aggregated = {key: sum(float(item[key]) for item in items if item.get(key) is not None) / len([item for item in items if item.get(key) is not None]) for key in keys}
    target_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    for item in items:
        summary = item.get("transition_summary", {})
        for key, value in summary.get("target_counts", {}).items():
            target_counts[key] = target_counts.get(key, 0) + int(value)
        for key, value in summary.get("predicted_counts", {}).items():
            predicted_counts[key] = predicted_counts.get(key, 0) + int(value)
    aggregated["transition_summary"] = {
        "target_counts": target_counts,
        "predicted_counts": predicted_counts,
    }
    return aggregated


def _evaluate(model, loader, *, device, num_classes: int) -> dict[str, Any]:
    import torch

    from land_change_detection.metrics.semantic_change import semantic_change_metrics

    model.eval()
    metrics: list[dict[str, Any]] = []
    samples = 0
    with torch.no_grad():
        for batch in loader:
            prepared = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            output = model(prepared["before"], prepared["after"])
            metrics.append(semantic_change_metrics(output, prepared, num_classes=num_classes))
            samples += int(prepared["before"].shape[0])
    payload = _aggregate_metrics(metrics)
    payload["samples"] = samples
    return payload


def _score(metrics: dict[str, Any]) -> tuple[float, float]:
    return (float(metrics.get("semantic_mean_iou", 0.0)), float(metrics.get("transition_mean_iou", 0.0)))


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from land_change_detection.models import SemanticChangeModelConfig, build_semantic_change_model
    from land_change_detection.training.losses import semantic_change_loss

    train_manifest = args.train_manifest or args.manifest
    val_manifest = args.val_manifest or args.manifest
    if train_manifest is None or val_manifest is None:
        raise SystemExit("Provide --train-manifest and --val-manifest, or use --manifest for smoke mode.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_supervised_dataset(train_manifest, args.input_channels)
    val_dataset = _load_supervised_dataset(val_manifest, args.input_channels)
    test_dataset = _load_supervised_dataset(args.test_manifest, args.input_channels) if args.test_manifest else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset is not None else None

    model = build_semantic_change_model(
        SemanticChangeModelConfig(
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    config = {
        "run_tag": args.run_tag,
        "task": "semantic_change_baseline_training",
        "baseline_pipeline": "T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable report",
        "project_role": "supervised semantic/transition baseline and future segmentation stage, not the full multimodal UniChange v2 model",
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "test_manifest": str(args.test_manifest) if args.test_manifest else None,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "input_channels": args.input_channels,
        "num_classes": args.num_classes,
        "base_channels": args.base_channels,
        "best_checkpoint_metric": "semantic_mean_iou then transition_mean_iou",
        "binary_change_role": "localization QA for the semantic baseline",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0)
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            prepared = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(prepared["before"], prepared["after"])
            loss = semantic_change_loss(
                output,
                prepared["before_target"],
                prepared["after_target"],
                change_target=prepared["change_target"],
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = _evaluate(model, val_loader, device=device, num_classes=args.num_classes)
        epoch_payload = {
            "epoch": epoch,
            "train_loss": _mean_loss(losses),
            "val": val_metrics,
        }
        history.append(epoch_payload)
        (output_dir / "metrics_history.json").write_text(json.dumps(_json_ready({"history": history}), indent=2), encoding="utf-8")
        torch.save(model.state_dict(), output_dir / "last.pt")

        score = _score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            (output_dir / "semantic_eval_val.json").write_text(json.dumps(_json_ready(val_metrics), indent=2), encoding="utf-8")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if not (output_dir / "best.pt").exists():
        torch.save(model.state_dict(), output_dir / "best.pt")
    shutil.copyfile(output_dir / "best.pt", output_dir / "semantic_change_model.pt")
    if not (output_dir / "semantic_eval_val.json").exists():
        val_metrics = _evaluate(model, val_loader, device=device, num_classes=args.num_classes)
        (output_dir / "semantic_eval_val.json").write_text(json.dumps(_json_ready(val_metrics), indent=2), encoding="utf-8")

    if test_loader is not None:
        state = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(state)
        test_metrics = _evaluate(model, test_loader, device=device, num_classes=args.num_classes)
        (output_dir / "semantic_eval_test.json").write_text(json.dumps(_json_ready(test_metrics), indent=2), encoding="utf-8")

    legacy_metrics = {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "samples": len(train_dataset),
        "mean_epoch_loss": _mean_loss([float(item["train_loss"]) for item in history]),
        "epoch_losses": [float(item["train_loss"]) for item in history],
        "input_channels": args.input_channels,
        "num_classes": args.num_classes,
        "run_tag": args.run_tag,
        "best_validation": json.loads((output_dir / "semantic_eval_val.json").read_text(encoding="utf-8")),
    }
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(legacy_metrics), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
