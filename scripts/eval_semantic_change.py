from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the standalone SemanticChangeBaseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
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


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from land_change_detection.metrics.semantic_change import semantic_change_metrics
    from land_change_detection.models import SemanticChangeModelConfig, build_semantic_change_model
    from land_change_detection.training.semantic_manifest_dataset import (
        SemanticManifestDataset,
        filter_semantic_supervised_records,
        load_semantic_manifest,
    )

    records = filter_semantic_supervised_records(load_semantic_manifest(args.manifest))
    if not records:
        raise RuntimeError(
            f"No semantic-transition supervised rows found in {args.manifest}. "
            "Evaluation requires semantic_before_path and semantic_after_path."
        )
    dataset = SemanticManifestDataset(records, input_channels=args.input_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_semantic_change_model(
        SemanticChangeModelConfig(
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
        )
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    metric_items: list[dict[str, Any]] = []
    samples = 0
    with torch.no_grad():
        for batch in loader:
            prepared = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            output = model(prepared["before"], prepared["after"])
            metric_items.append(semantic_change_metrics(output, prepared, num_classes=args.num_classes))
            samples += int(prepared["before"].shape[0])

    metrics = _aggregate_metrics(metric_items)
    metrics.update(
        {
            "samples": samples,
            "input_channels": args.input_channels,
            "num_classes": args.num_classes,
            "run_tag": args.run_tag,
            "baseline_pipeline": "T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable report",
            "project_role": "supervised semantic/transition baseline and future segmentation stage, not the full multimodal UniChange v2 model",
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
