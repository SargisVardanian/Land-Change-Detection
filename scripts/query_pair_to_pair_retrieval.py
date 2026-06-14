from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from scripts.train_dino_pair_retrieval import PairRetrievalDataset, choose_device, collate_batch, load_retrieval_samples
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query pair-to-pair retrieval and render top-k results.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-backbone", choices=("simple_patch", "dinov2"), default="simple_patch")
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _render_pair(before_path: str, after_path: str, size: tuple[int, int]) -> Image.Image:
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    before.thumbnail(size)
    after.thumbnail(size)
    canvas = Image.new("RGB", (size[0] * 2, size[1] + 28), (12, 12, 12))
    canvas.paste(before, (0, 28))
    canvas.paste(after, (size[0], 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "T1 / T2", fill=(255, 255, 255))
    return canvas


def main() -> int:
    args = parse_args()
    samples = load_retrieval_samples(None, list(args.manifest))
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if args.query_sample_id not in sample_by_id:
        raise SystemExit(f"Query sample not found: {args.query_sample_id}")
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
    dataset = PairRetrievalDataset(samples, args.image_size)
    batch = collate_batch([dataset[index] for index in range(len(dataset))])
    with torch.no_grad():
        outputs = model(batch["before"].to(device), batch["after"].to(device), None)
    embeddings = outputs["change_embedding"].detach().cpu()
    query_index = next(index for index, sample in enumerate(samples) if sample.sample_id == args.query_sample_id)
    ranking = torch.argsort(embeddings @ embeddings[query_index], descending=True).tolist()
    ranking = [index for index in ranking if index != query_index][: args.top_k]

    tile_size = (160, 160)
    canvas = Image.new("RGB", (tile_size[0] * 2, (tile_size[1] + 48) * (len(ranking) + 1)), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    query = sample_by_id[args.query_sample_id]
    canvas.paste(_render_pair(query.before_path, query.after_path, tile_size), (0, 0))
    draw.text((8, tile_size[1] + 32), f"QUERY {query.sample_id} | {query.transition_label}", fill=(255, 255, 255))
    for row_index, sample_index in enumerate(ranking, start=1):
        sample = samples[sample_index]
        y = row_index * (tile_size[1] + 48)
        canvas.paste(_render_pair(sample.before_path, sample.after_path, tile_size), (0, y))
        score = float(torch.dot(embeddings[query_index], embeddings[sample_index]))
        draw.text((8, y + tile_size[1] + 32), f"TOP{row_index} {sample.sample_id} | {sample.transition_label} | score={score:.4f}", fill=(255, 255, 255))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(json.dumps({"query_sample_id": args.query_sample_id, "top_k": [samples[index].sample_id for index in ranking], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
