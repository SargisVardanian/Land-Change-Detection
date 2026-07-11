from __future__ import annotations

import argparse
import json
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
from land_change_detection.models.qcpr import QCPRPatchReranker
from land_change_detection.temporal_caption_manifest import manifest_file_fingerprint
from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset, load_dataset_config, parse_dataset_weights
from land_change_detection.visualization import mask_rgba_overlay, rgb_absolute_difference
from ucv2_cluster_common import build_model, strict_device
from ucv2_retrieval_metrics import collect_retrieval_corpus, compute_retrieval_metrics, compute_retrieval_ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--dataset-weight", action="append", default=None)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _sample_paths(sample) -> tuple[str, str, str | None]:
    if isinstance(sample, dict):
        return str(sample["t1_path"]), str(sample["t2_path"]), sample.get("mask_path")
    return str(sample.image_before), str(sample.image_after), str(sample.binary_change_mask)


def render_pair(axes, sample, heading: str, score: float | None = None) -> None:
    t1_path, t2_path, mask_path = _sample_paths(sample)
    t1 = load_rgb(t1_path)
    t2 = load_rgb(t2_path)
    diff = rgb_absolute_difference(t1, t2)
    axes[0].imshow(t1)
    axes[0].set_title(f"{heading}: T1")
    axes[1].imshow(t2)
    axes[1].set_title("T2" if score is None else f"T2 | score={score:.4f}")
    axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("RGB difference")
    axes[3].imshow(t2)
    if mask_path:
        axes[3].imshow(mask_rgba_overlay(load_mask(mask_path)))
        axes[3].set_title("GT mask")
    else:
        axes[3].set_title("No binary mask")
    for axis in axes:
        axis.axis("off")


def _candidate_mask(logits: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    side = int(round(logits.numel() ** 0.5))
    if side * side != logits.numel():
        raise ValueError("QCPR patch grid must be square")
    return torch.nn.functional.interpolate(
        logits.sigmoid().reshape(1, 1, side, side), size=shape, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()


def _per_result_overlap(mask: np.ndarray, gt: np.ndarray | None, threshold: float) -> tuple[float | None, float | None]:
    if gt is None:
        return None, None
    predicted = mask >= threshold
    target = gt > 0
    intersection = int(np.logical_and(predicted, target).sum())
    denominator = int(predicted.sum() + target.sum())
    union = int(np.logical_or(predicted, target).sum())
    return (2.0 * intersection / denominator if denominator else 1.0, intersection / union if union else 1.0)


def _top_regions(mask: np.ndarray, count: int = 3) -> list[dict[str, float]]:
    flat = mask.reshape(-1)
    indices = np.argsort(flat)[::-1][:count]
    height, width = mask.shape
    return [
        {"x": float((index % width + 0.5) / width), "y": float((index // width + 0.5) / height), "probability": float(flat[index])}
        for index in indices
    ]


def _query_evidence(query: str, regions: list[dict[str, float]]) -> dict[str, object]:
    normalized = query.casefold()
    directions = [word for word in ("appeared", "disappeared", "changed", "increased", "decreased") if word in normalized]
    locations = [word for word in ("left", "right", "top", "bottom", "center", "middle") if word in normalized]
    return {
        "object_evidence": query,
        "temporal_direction_evidence": directions,
        "location_evidence": {"query_terms": locations, "mask_peak_regions": regions},
    }


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
            "val_manifests": tuple(str(path) for path in args.val_manifest),
            "dataset_config": str(args.dataset_config) if args.dataset_config else None,
            "dataset_sampling_weights": tuple(args.dataset_weight or saved.get("dataset_sampling_weights", ())),
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


def _eval_manifests(args: argparse.Namespace) -> tuple[list[str], dict[str, float]]:
    _, config_val, config_weights, _ = load_dataset_config(args.dataset_config)
    manifests = [str(path) for path in (config_val or [str(path) for path in args.val_manifest])]
    weights = config_weights or parse_dataset_weights(args.dataset_weight)
    return manifests, weights


def _build_eval_dataset(args: argparse.Namespace, config: SimpleNamespace):
    val_manifests, _ = _eval_manifests(args)
    if val_manifests:
        dataset = TemporalCaptionManifestDataset(
            val_manifests,
            split=args.split,
            image_size=int(config.image_size),
            output_grid=int(config.output_grid),
        )
        return dataset, "mixed"
    dataset = UniChangeMciDataset(
        args.data_root,
        split=args.split,
        image_size=int(config.image_size),
        output_grid=int(config.output_grid),
    )
    return dataset, "levir_only"


def _eval_metadata(args: argparse.Namespace, dataset, data_mode: str) -> dict[str, object]:
    val_manifests, weights = _eval_manifests(args)
    if val_manifests:
        dataset_names = sorted(getattr(dataset, "indices_by_dataset", {}).keys())
        row_counts = {str(name): len(indices) for name, indices in sorted(dataset.indices_by_dataset.items())}
    else:
        dataset_names = ["levir_mci"]
        row_counts = {"levir_mci": len(dataset)}
        weights = {}
    return {
        "data_mode": data_mode,
        "manifest_fingerprints": {"validation": {path: manifest_file_fingerprint(path) for path in val_manifests}},
        "dataset_names": dataset_names,
        "dataset_weights": weights if data_mode == "mixed" else {},
        "validation_row_counts": row_counts,
        "validation_row_count": len(dataset),
    }


def _sample_map(dataset) -> dict[str, object]:
    samples = getattr(dataset, "samples", [])
    mapping: dict[str, object] = {}
    for sample in samples:
        if isinstance(sample, dict):
            mapping[str(sample["pair_id"])] = sample
        else:
            mapping[str(sample.sample_id)] = sample
    return mapping


def _query_items(args: argparse.Namespace, dataset, pair_index: dict[str, int]) -> list[dict[str, object]]:
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        return [
            item
            for item in manifest.get("items", [])
            if item.get("pair_id") in pair_index
        ][: args.max_queries]
    items: list[dict[str, object]] = []
    for sample in getattr(dataset, "samples", []):
        if isinstance(sample, dict):
            captions = sample.get("captions", [])
            pair_id = str(sample["pair_id"])
        else:
            captions = sample.captions or ([sample.caption] if sample.caption else [])
            pair_id = str(sample.sample_id)
        for caption in captions[:1]:
            if pair_id in pair_index:
                items.append({"pair_id": pair_id, "query_caption": caption, "stratum": "canonical_manifest"})
        if len(items) >= args.max_queries:
            break
    return items


def main() -> int:
    args = parse_args()
    device = strict_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = _checkpoint_config(args, checkpoint)
    dataset, data_mode = _build_eval_dataset(args, config)
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
    metrics, similarities = compute_retrieval_metrics(
        corpus, query_chunk_size=args.query_chunk_size, candidate_chunk_size=args.candidate_chunk_size
    )
    global_similarities = corpus.text_embeddings.float() @ corpus.pair_embeddings.float().T
    local_similarities = None
    if corpus.patch_tokens is not None and corpus.mask_query_embeddings is not None:
        if corpus.qcpr_beta > 0:
            local_similarities = (similarities - corpus.qcpr_alpha * global_similarities) / corpus.qcpr_beta
        else:
            local_similarities = torch.zeros_like(similarities)
    rank_result = compute_retrieval_ranks(
        corpus, query_chunk_size=args.query_chunk_size, candidate_chunk_size=args.candidate_chunk_size
    )

    pair_index = {pair_id: index for index, pair_id in enumerate(corpus.pair_ids)}
    sample_by_id = _sample_map(dataset)
    query_items = _query_items(args, dataset, pair_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals = args.output_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "qcpr_mask_overlays"
    if local_similarities is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
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
        relevant_mask = rank_result.positive_mask[caption_index]
        scores = similarities[caption_index]
        global_scores = global_similarities[caption_index]
        local_scores = local_similarities[caption_index] if local_similarities is not None else None
        order = rank_result.ranked_candidate_indices[caption_index]
        relevant_rank = int(rank_result.duplicate_aware_ranks[caption_index].item())
        exact_rank = int(rank_result.exact_pair_ranks[caption_index].item())
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
            global_score = float(global_scores[int(retrieved_index)].item())
            local_score = float(local_scores[int(retrieved_index)].item()) if local_scores is not None else None
            is_relevant = bool(relevant_mask[int(retrieved_index)].item())
            is_exact = int(retrieved_index) == query_pair_index
            label = f"#{row} {retrieved_id} {'RELEVANT' if is_relevant else 'OTHER'}"
            render_pair(axes[row], sample_by_id[retrieved_id], label, score)
            result_record = {
                    "rank": row,
                    "pair_id": retrieved_id,
                    "score": score,
                    "S_global": global_score,
                    "S_local": local_score,
                    "S_final": score,
                    "global_score": global_score,
                    "local_score": local_score,
                    "final_score": score,
                    "score_mode": corpus.score_mode,
                    "semantic_relevance": is_relevant,
                    "exact_pair_relevance": is_exact,
                }
            if corpus.patch_tokens is not None and corpus.mask_query_embeddings is not None:
                candidate_patches = corpus.patch_tokens[int(retrieved_index)].float()
                mask_query = corpus.mask_query_embeddings[caption_index].float()
                patch_logits = torch.einsum("d,nd->n", mask_query, candidate_patches)
                pooled = QCPRPatchReranker.masked_local_embedding(
                    candidate_patches.unsqueeze(0), patch_logits.reshape(1, 1, -1)
                )
                recomputed_local = float(torch.einsum(
                    "d,qbd->qb", corpus.text_embeddings[caption_index].float(), pooled
                )[0, 0].item())
                if local_score is None or not np.isclose(recomputed_local, local_score, atol=1e-5):
                    raise RuntimeError("Displayed candidate mask logits do not reproduce S_local")
                t1_path, t2_path, gt_path = _sample_paths(sample_by_id[retrieved_id])
                t1, t2 = load_rgb(t1_path), load_rgb(t2_path)
                soft = _candidate_mask(patch_logits, t2.shape[:2])
                thresholded = soft >= args.mask_threshold
                stem = f"query_{query_number:03d}_rank_{row:02d}_{retrieved_id}"
                soft_path = overlay_dir / f"{stem}_soft.png"
                threshold_path = overlay_dir / f"{stem}_thresholded.png"
                Image.fromarray(np.uint8(np.clip(soft, 0, 1) * 255), mode="L").save(soft_path)
                Image.fromarray(np.uint8(thresholded) * 255, mode="L").save(threshold_path)
                gt = load_mask(gt_path) if gt_path else None
                dice, iou = _per_result_overlap(soft, gt, args.mask_threshold)
                regions = _top_regions(soft)
                evidence = _query_evidence(query_caption, regions)
                panel, panel_axes = plt.subplots(2, 4, figsize=(18, 9))
                panel_axes[0, 0].imshow(t1); panel_axes[0, 0].set_title("T1")
                panel_axes[0, 1].imshow(t2); panel_axes[0, 1].set_title("T2")
                panel_axes[0, 2].imshow(rgb_absolute_difference(t1, t2), cmap="magma"); panel_axes[0, 2].set_title("RGB difference")
                panel_axes[0, 3].imshow(gt if gt is not None else np.zeros(t2.shape[:2]), cmap="gray"); panel_axes[0, 3].set_title("GT mask" if gt is not None else "GT unavailable")
                panel_axes[1, 0].imshow(t1); panel_axes[1, 0].imshow(soft, cmap="magma", alpha=.55, vmin=0, vmax=1); panel_axes[1, 0].set_title("Soft mask over T1")
                panel_axes[1, 1].imshow(t2); panel_axes[1, 1].imshow(soft, cmap="magma", alpha=.55, vmin=0, vmax=1); panel_axes[1, 1].set_title("Soft mask over T2")
                panel_axes[1, 2].imshow(thresholded, cmap="gray"); panel_axes[1, 2].set_title(f"Thresholded @{args.mask_threshold:g}")
                panel_axes[1, 3].axis("off"); panel_axes[1, 3].text(0, .95, f"pair_id: {retrieved_id}\nS_global={global_score:.4f}\nS_local={local_score:.4f}\nS_final={score:.4f}\nsemantic={is_relevant}\nexact={is_exact}", va="top")
                for axis in panel_axes.flat[:7]: axis.axis("off")
                panel.suptitle(f"Exact query: {query_caption}")
                panel.tight_layout(rect=(0, 0, 1, .96))
                panel_path = visuals / f"{stem}.png"
                panel.savefig(panel_path, dpi=140, bbox_inches="tight"); plt.close(panel)
                result_record.update({
                    "query": query_caption, "soft_mask_path": str(soft_path),
                    "thresholded_mask_path": str(threshold_path), "explanation_panel_path": str(panel_path),
                    "per_result_Dice": dice, "per_result_IoU": iou, "top_regions": regions,
                    **evidence,
                })
                if corpus.temporal_explanation_logits is not None:
                    channels = corpus.temporal_explanation_logits[int(retrieved_index)].sigmoid().T
                    channel_paths = {}
                    for channel_index, channel_name in enumerate(("appeared", "disappeared", "changed")):
                        channel_mask = _candidate_mask(
                            torch.logit(channels[channel_index].clamp(1e-6, 1 - 1e-6)), t2.shape[:2]
                        )
                        channel_path = overlay_dir / f"{stem}_{channel_name}.png"
                        Image.fromarray(np.uint8(channel_mask * 255), mode="L").save(channel_path)
                        channel_paths[f"{channel_name}_probability_map_path"] = str(channel_path)
                    result_record.update(channel_paths)
            retrieved.append(result_record)
        figure.suptitle(
            f"Query: {query_caption}\nRelevant rank={relevant_rank}; exact pair rank={exact_rank}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        image_name = f"query_{query_number:03d}_{query_pair_id}.png"
        figure.savefig(visuals / image_name, dpi=140, bbox_inches="tight")
        plt.close(figure)
        top_global = float(global_scores[int(top_indices[0])].item()) if top_indices else None
        top_local = float(local_scores[int(top_indices[0])].item()) if local_scores is not None and top_indices else None
        top_final = float(scores[int(top_indices[0])].item()) if top_indices else None
        records.append(
            {
                "query_pair_id": query_pair_id,
                "query_caption": query_caption,
                "stratum": item.get("stratum"),
                "best_relevant_rank": relevant_rank,
                "exact_pair_rank": exact_rank,
                "retrieved": retrieved,
                "image": image_name,
                "mask_overlay": retrieved[0].get("explanation_panel_path") if retrieved else None,
                "S_global": top_global,
                "S_local": top_local,
                "S_final": top_final,
                "global_score": top_global,
                "local_score": top_local,
                "final_score": top_final,
                "score_mode": corpus.score_mode,
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
        **_eval_metadata(args, dataset, data_mode),
        "corpus_metrics": metrics,
        "gallery_metrics": gallery_metrics,
        "top_k": args.top_k,
        "patch_reranker_available": corpus.patch_tokens is not None,
        "qcpr_score_mode": corpus.score_mode,
        **{
            key: value
            for key, value in metrics.items()
            if key.startswith("mask_") or key in {"predicted_mask_area_mean", "target_mask_area_mean"}
        },
    }
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "retrieval_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
