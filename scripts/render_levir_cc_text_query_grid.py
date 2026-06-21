from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from scripts.train_dino_pair_retrieval import PairRetrievalDataset, choose_device, collate_batch, load_retrieval_samples
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.retrieval_baselines import preset_names
from scripts.train_dino_pair_retrieval import apply_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render top-k qualitative text-to-pair retrieval grids for LEVIR-CC queries.")
    parser.add_argument("--preset", choices=preset_names(), default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-backbone", choices=("simple_patch", "dinov2"), default="simple_patch")
    parser.add_argument("--text-backbone", choices=("simple_text", "remoteclip", "hf_remoteclip", "openclip"), default="remoteclip")
    parser.add_argument("--pair-feature-mode", choices=("t2_only", "signed_delta", "change_fusion"), default="change_fusion")
    parser.add_argument("--dinov2-model-path", type=Path, default=None)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--remoteclip-checkpoint", type=Path, default=None)
    parser.add_argument("--hf-remoteclip-model-path", type=Path, default=None)
    parser.add_argument("--openclip-model-name", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def _render_pair(before_path: str, after_path: str, size: tuple[int, int]) -> Image.Image:
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    before.thumbnail(size)
    after.thumbnail(size)
    canvas = Image.new("RGB", (size[0] * 2, size[1] + 26), (15, 15, 15))
    canvas.paste(before, (0, 26))
    canvas.paste(after, (size[0], 26))
    ImageDraw.Draw(canvas).text((8, 6), "T1 / T2", fill=(255, 255, 255))
    return canvas


def main() -> int:
    args = parse_args()
    args, _preset_payload = apply_preset(args)
    samples = load_retrieval_samples(args.manifest, [], args.project_root)
    caption_samples = [sample for sample in samples if sample.caption]
    if not caption_samples:
        raise SystemExit("Manifest contains no caption rows.")
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone=args.visual_backbone,
            text_backbone=args.text_backbone,
            pair_feature_mode=args.pair_feature_mode,
            dinov2_model_path=str(args.dinov2_model_path) if args.dinov2_model_path else None,
            remoteclip_arch=args.remoteclip_arch,
            remoteclip_checkpoint=str(args.remoteclip_checkpoint) if args.remoteclip_checkpoint else None,
            hf_remoteclip_model_path=str(args.hf_remoteclip_model_path) if args.hf_remoteclip_model_path else None,
            openclip_model_name=args.openclip_model_name,
            openclip_pretrained=args.openclip_pretrained,
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
        pair_embeddings = outputs["change_embedding"].detach().cpu()
        query_rows = caption_samples[: args.num_queries]
        text_embeddings = model._encode_text([sample.caption or "" for sample in query_rows], device).detach().cpu()

    unique_pairs: OrderedDict[str, int] = OrderedDict()
    for index, sample in enumerate(samples):
        unique_pairs.setdefault(sample.pair_id, index)
    unique_pair_ids = list(unique_pairs.keys())
    unique_pair_indices = list(unique_pairs.values())
    unique_pair_embeddings = pair_embeddings[unique_pair_indices]
    pair_lookup = {pair_id: samples[index] for pair_id, index in unique_pairs.items()}

    tile_size = (120, 120)
    row_height = tile_size[1] + 86
    width = tile_size[0] * 2 * (args.top_k + 1)
    height = row_height * len(query_rows)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    report_rows = []

    for row_index, query_sample in enumerate(query_rows):
        y = row_index * row_height
        query_embedding = text_embeddings[row_index]
        scores = unique_pair_embeddings @ query_embedding
        ranking = torch.argsort(scores, descending=True).tolist()[: args.top_k]
        query_pair = pair_lookup[query_sample.pair_id]
        canvas.paste(_render_pair(query_pair.before_path, query_pair.after_path, tile_size), (0, y))
        draw.text((8, y + tile_size[1] + 34), f"Q{row_index+1}: {query_sample.caption}", fill=(255, 255, 255))
        draw.text((8, y + tile_size[1] + 52), f"pair_id={query_sample.pair_id}", fill=(180, 180, 180))
        ranked_pairs = []
        for rank, candidate_index in enumerate(ranking, start=1):
            pair_id = unique_pair_ids[candidate_index]
            candidate = pair_lookup[pair_id]
            x = rank * tile_size[0] * 2
            canvas.paste(_render_pair(candidate.before_path, candidate.after_path, tile_size), (x, y))
            score = float(scores[candidate_index].item())
            draw.text((x + 8, y + tile_size[1] + 34), f"TOP{rank} {pair_id}", fill=(255, 255, 255))
            draw.text((x + 8, y + tile_size[1] + 52), f"score={score:.4f}", fill=(180, 180, 180))
            ranked_pairs.append({"pair_id": pair_id, "score": score})
        report_rows.append({"query_caption": query_sample.caption, "query_pair_id": query_sample.pair_id, "top_k": ranked_pairs})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    report = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "num_queries": len(query_rows),
        "top_k": args.top_k,
        "output": str(args.output),
        "queries": report_rows,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
