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
from land_change_detection.data.unichange_subset import resolve_levir_mci_root
from land_change_detection.metrics.unichange_retrieval import dataset_retrieval_metrics, write_full_rankings
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-level evaluation for UniChange joint overfit.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_levir_mci_root(args.data_root).root
    dataset = UniChangeMciDataset(root, split="train", image_size=224, subset_file=args.subset_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_unichange_mci)
    model = UniChangeModel(
        UniverSatJointBackend(UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint)),
        JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=64, freeze=True)),
        UniChangeConfig(event_queries=16),
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint.get("heads", checkpoint.get("model", {})), strict=False)
    model.eval()
    pair_ids: list[str] = []
    captions: list[str] = []
    caption_to_pair_parts: list[torch.Tensor] = []
    pair_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["t1"].to(args.device), batch["t2"].to(args.device), batch["captions"], temporal_context=batch["temporal_context"])
            pair_ids.extend(batch["pair_ids"])
            captions.extend(batch["captions"])
            pair_embeddings.append(out.global_pair_embedding.cpu())
            text_embeddings.append(out.text_global_embedding.cpu())  # type: ignore[union-attr]
            caption_to_pair_parts.append(batch["caption_to_pair"].cpu() + offset)
            offset += len(batch["pair_ids"])
    pairs = torch.cat(pair_embeddings, dim=0)
    texts = torch.cat(text_embeddings, dim=0)
    caption_to_pair = torch.cat(caption_to_pair_parts, dim=0)
    metrics = dataset_retrieval_metrics(pairs, texts, caption_to_pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings = args.output.parent / "rankings.jsonl"
    write_full_rankings(str(rankings), pair_ids, captions, texts @ pairs.T)
    args.output.write_text(json.dumps({"metrics": metrics, "rankings": str(rankings), "subset_file": str(args.subset_file)}, indent=2))


if __name__ == "__main__":
    main()
