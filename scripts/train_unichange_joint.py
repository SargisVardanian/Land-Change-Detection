#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from torch.utils.data import DataLoader

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.data.unichange_mci import UniChangeMciDataset, collate_unichange_mci
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel
from land_change_detection.training.unichange_joint_trainer import UniChangeJointTrainer, UniChangeJointTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified UniChange retrieval-grounding milestone.")
    root = Path(os.environ.get("RS_PROJECT_ROOT", Path.cwd()))
    parser.add_argument("--data-root", type=Path, default=root / "datasets" / "processed" / "LEVIR-MCI")
    parser.add_argument("--universat-source", type=Path, default=root / "external" / "UniverSat")
    parser.add_argument("--universat-checkpoint", type=Path, default=root / "models" / "universat-base")
    parser.add_argument("--jina-model", type=Path, default=root / "models" / "jina-v5-text-small-retrieval")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "joint_retrieval_grounding_overfit_100")
    parser.add_argument("--run-name", default="joint_retrieval_grounding_overfit_100")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = UniChangeMciDataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        output_grid=36,
        min_component_area=args.min_component_area,
        max_pairs=args.max_pairs,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No LEVIR-MCI samples found under {args.data_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_unichange_mci)
    visual = UniverSatJointBackend(UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint))
    text = JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=64, freeze=True))
    model = UniChangeModel(visual_encoder=visual, text_encoder=text, config=UniChangeConfig(event_queries=16))
    trainer = UniChangeJointTrainer(
        model,
        UniChangeJointTrainerConfig(
            run_name=args.run_name,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            total_steps=args.total_steps,
            device=args.device,
        ),
    )
    metrics = trainer.fit(loader)
    manifest = {
        "run_name": args.run_name,
        "dataset_size": len(dataset),
        "max_pairs": args.max_pairs,
        "temporal_context": {
            "order": ["before", "after"],
            "delta_days": None,
            "duration_known": False,
            "time_semantics": "ordinal_not_calendar",
        },
        "final_metrics": metrics,
        "overfit_gates": {
            "requires_ysu_slurm_logs": True,
            "R@5": 0.95,
            "R@10": 0.99,
            "union_mask_dice": 0.85,
            "text_conditioned_energy_inside_gt": 0.60,
        },
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
