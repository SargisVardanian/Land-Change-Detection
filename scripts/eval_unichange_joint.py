#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.data.unichange_mci import UniChangeMciDataset, collate_unichange_mci
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel
from land_change_detection.training.unichange_joint_trainer import UniChangeJointTrainer, UniChangeJointTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a UniChange joint overfit checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = UniChangeMciDataset(args.data_root, split="train", image_size=224, max_pairs=args.max_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_unichange_mci)
    model = UniChangeModel(
        UniverSatJointBackend(UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint)),
        JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=64, freeze=True)),
        UniChangeConfig(event_queries=16),
    )
    trainer = UniChangeJointTrainer(model, UniChangeJointTrainerConfig(output_dir=args.output.parent, epochs=1, device=args.device))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            result = trainer.compute_loss(batch, step=index, total_steps=max(len(loader), 1))
            rows.append(result.metrics)
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "batches": rows}, indent=2))


if __name__ == "__main__":
    main()
