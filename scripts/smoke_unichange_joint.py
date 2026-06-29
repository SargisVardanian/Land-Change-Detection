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
from land_change_detection.data.unichange_subset import build_deterministic_mci_subset, resolve_levir_mci_root, validate_mci_subset
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel
from land_change_detection.training.unichange_joint_trainer import UniChangeJointTrainer, UniChangeJointTrainerConfig
from scripts.render_unichange_joint import render_training_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real GPU smoke for UniChange joint training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_levir_mci_root(args.data_root).root
    if not args.subset_file.exists():
        build_deterministic_mci_subset(root, args.subset_file, count=2, split="train")
    else:
        validate_mci_subset(args.subset_file, expected_count=2, expected_split="train")
    dataset = UniChangeMciDataset(root, split="train", image_size=224, subset_file=args.subset_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_unichange_mci)
    model = UniChangeModel(
        UniverSatJointBackend(UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint)),
        JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=64, freeze=True)),
        UniChangeConfig(event_queries=16),
    )
    trainer = UniChangeJointTrainer(model, UniChangeJointTrainerConfig(output_dir=args.output_dir, total_steps=1, device=args.device, use_bf16=args.use_bf16))
    batch = next(iter(loader))
    metrics = trainer.train_step(batch, step=0, total_steps=1)
    checkpoint = args.output_dir / "smoke_heads.pt"
    trainer.save_checkpoint(checkpoint, 0, metrics)
    trainer.load_checkpoint(checkpoint)
    out = trainer.model(batch["t1"].to(trainer.device), batch["t2"].to(trainer.device), batch["captions"], temporal_context=batch["temporal_context"])
    panel = args.output_dir / "smoke_panel.png"
    render_training_panel(
        panel,
        batch["t1"][0],
        batch["t2"][0],
        batch["masks"][0],
        out.event_masks[0].detach().cpu().max(dim=0).values.view(36, 36),  # type: ignore[index]
        out.event_masks[0].detach().cpu().view(-1, 36, 36)[:4],  # type: ignore[index]
        out.text_conditioned_mask[0, 0].detach().cpu().view(36, 36),  # type: ignore[index]
        batch["captions"][0],
        ["two-pair retrieval smoke"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_report.json").write_text(json.dumps({"metrics": metrics, "checkpoint": str(checkpoint), "panel": str(panel)}, indent=2))


if __name__ == "__main__":
    main()
