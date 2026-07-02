from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.data.unichange_mci import UniChangeMciDataset
from land_change_detection.models.retrieval_heads import normalize_caption_text
from land_change_detection.visualization import mask_rgba_overlay, rgb_absolute_difference
from ucv2_cluster_common import build_model, strict_device
from ucv2_retrieval_metrics import collect_retrieval_corpus, compute_retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def render_pair(axes, sample, heading: str, score: float | None = None) -> None:
    t1 = load_rgb(sample.image_before)
    t2 = load_rgb(sample.image_after)
    mask = load_mask(sample.binary_change_mask)
    diff = rgb_absolute_difference(t1, t2)
    axes[0].imshow(t1)
    axes[0].set_title(f"{heading}: T1")
    axes[1].imshow(t2)
    axes[1].set_title("T2" if score is None else f"T2 | score={score:.4f}")
    axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("RGB difference")
    axes[3].imshow(t2)
    axes[3].imshow(mask_rgba_overlay(mask))
    axes[3].set_title("GT mask")
    for axis in axes:
        axis.axis("off")


def _checkpoint_config(args: argparse.Namespace, checkpoint: dict) -> SimpleNamespace:
    saved = dict(checkpoint.get("config", {}))
    saved.update(
        {
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "universat_source": str(args.universat_source),
            "universat_checkpoint": str(args.universat_checkpoint),
            "jina_model": str(args.jina_model),
            "train_split": "train",
            "val_split": args.split,
            "batch_size": args.batch_size,
            "epochs": 1,
            "num_workers": args.num_workers,
            "use_bf16": True,
            "device": args.device,
        }
    )
    saved.setdefault("image_size", 256)
    saved.setdefault("output_grid", 32)
    saved.setdefault("temporal_depth", 4)
    saved.setdefault("use_direction_embeddings", False)
    saved.setdefault("use_explicit_change_fusion", False)
    saved.setdefault("trainable_temperature", False)
    saved.setdefault("initial_temperature", 0.07)
    saved.setdefault("max_logit_scale", 100.0)
    saved.setdefault("caption_frequency_power", 0.5)
    saved.setdefault("seed", 20260701)
    return SimpleNamespace(**saved)


def main() -> int:
    args = parse_args()
    device = strict_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = _checkpoint_config(args, checkpoint)
    image_size = int(config.image_size)
    output_grid = int(config.output_grid)
    dataset = UniChangeMciDataset(
        args.data_root,
        split=args.split,
        image_size=image_size,
        output_grid=output_grid,
    )
    if checkpoint.get("stage1_next"):
        from ucv2_stage1_next_core import make_eval_loader

        loader = make_eval_loader(dataset, config)
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=base._collate_temporal,
            pin_memory=device.type == "cuda",
        )
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    corpus = collect_retrieval_corpus(model, loader, device, config)
    metrics, similarities = compute_retrieval_metrics(corpus)

    pair_index = {pair_id: index for index, pair_id in enumerate(corpus.pair_ids)}
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    group_to_pairs: dict[int, set[int]] = defaultdict(set)
    for group_id, mapped_pair in zip(
        corpus.caption_group_ids.tolist(),
        corpus.caption_to_pair.tolist(),
        strict=True,
    ):
        group_to_pairs[int(group_id)].add(int(mapped_pair))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    query_items = [
        item
        for item in manifest.get("items", [])
        if item.get("pair_id") in pair_index
    ][: args.max_queries]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals = args.output_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    records = []
    relevance_ranks = []
    exact_ranks = []

    for query_number, item in enumerate(query_items):
        query_pair_id = str(item["pair_id"])
        query_pair_index = pair_index[query_pair_id]
        query_caption = str(item["query_caption"])
        normalized_query = normalize_caption_text(query_caption)
        caption_candidates = [
            index
            for index, (caption, mapped_pair) in enumerate(
                zip(corpus.captions, corpus.caption_to_pair.tolist(), strict=True)
            )
            if int(mapped_pair) == query_pair_index
            and normalize_caption_text(caption) == normalized_query
        ]
        if not caption_candidates:
            raise RuntimeError(f"Missing query caption for {query_pair_id}")
        caption_index = caption_candidates[0]
        relevant = group_to_pairs[int(corpus.caption_group_ids[caption_index].item())]
        scores = similarities[caption_index]
        order = torch.argsort(scores, descending=True)
        inverse_rank = torch.empty_like(order)
        inverse_rank[order] = torch.arange(order.numel())
        relevant_rank = min(int(inverse_rank[index].item()) + 1 for index in relevant)
        exact_rank = int(inverse_rank[query_pair_index].item()) + 1
        relevance_ranks.append(relevant_rank)
        exact_ranks.append(exact_rank)
        top_indices = order[: min(args.top_k, order.numel())].tolist()

        figure, axes = plt.subplots(
            1 + len(top_indices),
            4,
            figsize=(16, 4 * (1 + len(top_indices))),
        )
        if axes.ndim == 1:
            axes = axes[None, :]
        render_pair(axes[0], sample_by_id[query_pair_id], f"QUERY {query_pair_id}")
        retrieved = []
        for row, retrieved_index in enumerate(top_indices, start=1):
            retrieved_id = corpus.pair_ids[int(retrieved_index)]
            score = float(scores[int(retrieved_index)].item())
            is_relevant = int(retrieved_index) in relevant
            label = f"#{row} {retrieved_id} {'RELEVANT' if is_relevant else 'OTHER'}"
            render_pair(axes[row], sample_by_id[retrieved_id], label, score)
            retrieved.append(
                {
                    "rank": row,
                    "pair_id": retrieved_id,
                    "score": score,
                    "relevant": is_relevant,
                }
            )
        figure.suptitle(
            f"Query: {query_caption}\nRelevant rank={relevant_rank}; exact pair rank={exact_rank}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        image_name = f"query_{query_number:03d}_{query_pair_id}.png"
        figure.savefig(visuals / image_name, dpi=140, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {
                "query_pair_id": query_pair_id,
                "query_caption": query_caption,
                "stratum": item.get("stratum"),
                "best_relevant_rank": relevant_rank,
                "exact_pair_rank": exact_rank,
                "retrieved": retrieved,
                "image": image_name,
            }
        )

    rank_tensor = torch.tensor(relevance_ranks, dtype=torch.float32) if relevance_ranks else torch.empty(0)
    exact_tensor = torch.tensor(exact_ranks, dtype=torch.float32) if exact_ranks else torch.empty(0)
    gallery_metrics = {
        "query_count": len(records),
        "R@1": float((rank_tensor <= 1).float().mean().item()) if relevance_ranks else 0.0,
        "R@5": float((rank_tensor <= 5).float().mean().item()) if relevance_ranks else 0.0,
        "R@10": float((rank_tensor <= 10).float().mean().item()) if relevance_ranks else 0.0,
        "MRR": float((1.0 / rank_tensor).mean().item()) if relevance_ranks else 0.0,
        "exact_pair_R@1": float((exact_tensor <= 1).float().mean().item()) if exact_ranks else 0.0,
        "exact_pair_R@5": float((exact_tensor <= 5).float().mean().item()) if exact_ranks else 0.0,
        "exact_pair_R@10": float((exact_tensor <= 10).float().mean().item()) if exact_ranks else 0.0,
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "stage1_next": bool(checkpoint.get("stage1_next")),
        "checkpoint_epoch": checkpoint.get("epoch", checkpoint.get("epoch_index")),
        "checkpoint_step": checkpoint.get("step"),
        "split": args.split,
        "corpus_metrics": metrics,
        "gallery_metrics": gallery_metrics,
        "top_k": args.top_k,
    }
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "retrieval_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
