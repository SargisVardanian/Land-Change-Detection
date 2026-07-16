from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from land_change_detection.data.unichange_mci import UniChangeMciDataset
from land_change_detection.models.retrieval_heads import normalize_caption_text, semantic_teacher_relevance_matrix
from land_change_detection.models.qcpr import QCPRPatchReranker, _structured_signature, structured_hard_negative_masks
from land_change_detection.run_metadata import file_sha256
from land_change_detection.temporal_caption_manifest import manifest_file_fingerprint
from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset, load_dataset_config, parse_dataset_weights
from land_change_detection.visualization import mask_rgba_overlay, rgb_absolute_difference
from ucv2_cluster_common import build_model, strict_device
from ucv2_retrieval_metrics import (
    collect_retrieval_corpus,
    compute_retrieval_metrics,
    compute_retrieval_ranks,
    retrieval_branch_diagnostics,
    retrieval_branch_similarity_matrices,
)
from ucv2_progress import write_progress


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
    parser.add_argument("--rerank-top-n", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--progress-path", type=Path, default=None)
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


def _assert_no_nonfinite(value, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite evaluation artifact value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_nonfinite(item, f"{path}[{index}]")


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
    )[0, 0].detach().cpu().numpy()


def temporal_channel_render_status(checkpoint: dict, architecture_version: str) -> dict[str, object]:
    if architecture_version == "v1":
        return {
            "temporal_channels_available": False,
            "temporal_channels_trained": False,
            "label": "Unavailable for QCPR v1",
            "supervision_provenance": None,
            "gradient_provenance": None,
        }
    provenance = checkpoint.get("temporal_channel_provenance", {})
    trained = bool(checkpoint.get("temporal_channels_trained", False))
    return {
        "temporal_channels_available": True,
        "temporal_channels_trained": trained,
        "label": "Temporal channels" if trained else "Untrained for QCPR v2",
        "supervision_provenance": provenance.get("supervision"),
        "gradient_provenance": provenance.get("gradients"),
    }


def render_temporal_channel_panels(axes, channel_maps: dict[str, np.ndarray], status: dict[str, object]) -> None:
    can_explain = bool(status["temporal_channels_available"] and status["temporal_channels_trained"])
    for column, channel_name in enumerate(("changed", "appeared", "disappeared")):
        if can_explain:
            axes[column].imshow(channel_maps[channel_name], cmap="magma", vmin=0, vmax=1)
            axes[column].set_title(f"{channel_name} probability")
        else:
            axes[column].set_facecolor("#b8b8b8")
            axes[column].text(0.5, 0.5, str(status["label"]), ha="center", va="center", wrap=True)
            axes[column].set_title(channel_name)
        axes[column].axis("off")


def _result_logits(corpus, query_index: int, candidate_index: int) -> torch.Tensor:
    patches = corpus.patch_tokens[candidate_index : candidate_index + 1].float()
    if corpus.qcpr_architecture_version == "v1":
        return torch.einsum("d,nd->n", corpus.mask_query_embeddings[query_index].float(), patches[0])
    reranker = corpus.qcpr_reranker
    device = next(reranker.parameters()).device
    descriptor = patches.to(device)
    token = torch.nn.functional.normalize(reranker.token_projection(corpus.text_token_embeddings[query_index : query_index + 1].to(device)), dim=-1)
    attention = corpus.text_attention_mask[query_index : query_index + 1].to(device)
    affinity = torch.einsum("bnd,qld->qbnl", descriptor, token) / reranker.logit_scale
    affinity = affinity.masked_fill(~attention[:, None, None, :].bool(), -1e4)
    attended = torch.einsum("qbnl,qld->qbnd", affinity.softmax(-1), token)
    expanded = descriptor.unsqueeze(0)
    logits = reranker.interaction_mlp(torch.cat((expanded, attended, expanded * attended), dim=-1)).squeeze(-1)
    # Keep the panel logits identical to score_v2 and the chunked evaluator:
    # text interaction is a residual over generic changed-channel evidence.
    if corpus.temporal_explanation_logits is not None:
        logits = logits + corpus.temporal_explanation_logits[candidate_index : candidate_index + 1, :, 0].to(device).unsqueeze(0)
    return logits.squeeze().detach().cpu()


def _per_result_overlap(mask: np.ndarray, gt: np.ndarray | None, threshold: float) -> tuple[float | None, float | None]:
    if gt is None:
        return None, None
    predicted = mask >= threshold
    target = gt > 0
    intersection = int(np.logical_and(predicted, target).sum())
    denominator = int(predicted.sum() + target.sum())
    union = int(np.logical_or(predicted, target).sum())
    return (2.0 * intersection / denominator if denominator else 1.0, intersection / union if union else 1.0)


def _top_regions(mask: np.ndarray, threshold: float = 0.5, count: int = 5) -> list[dict[str, object]]:
    height, width = mask.shape
    labels, component_count = ndimage.label(mask >= threshold)
    regions: list[dict[str, object]] = []
    for label_index in range(1, component_count + 1):
        yy, xx = np.where(labels == label_index)
        if yy.size < max(4, int(0.001 * height * width)):
            continue
        regions.append({
            "normalized_bbox": [float(xx.min() / width), float(yy.min() / height), float((xx.max() + 1) / width), float((yy.max() + 1) / height)],
            "region_confidence": float(mask[yy, xx].mean()), "area_fraction": float(yy.size / (height * width)),
        })
    return sorted(regions, key=lambda item: float(item["region_confidence"]), reverse=True)[:count]


def _model_evidence(mask: np.ndarray, regions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "object_evidence": {"query_conditioned_peak": float(mask.max()), "query_conditioned_mean": float(mask.mean())},
        "temporal_direction_evidence": None,
        "location_evidence": {"connected_components": regions},
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
    evaluation_started = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_path or args.output_dir / "progress.json"
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
    temporal_channel_status = temporal_channel_render_status(checkpoint, config.qcpr_architecture_version)
    model.eval()
    corpus = collect_retrieval_corpus(
        model,
        loader,
        device,
        config,
        progress_path=progress_path,
        progress_stage="evaluation_encoding",
    )
    write_progress(progress_path, stage="evaluation_scoring", completed=0, total=1, started=time.perf_counter())
    branch_scores = retrieval_branch_similarity_matrices(
        corpus, query_chunk_size=args.query_chunk_size, candidate_chunk_size=args.candidate_chunk_size,
        rerank_top_n=args.rerank_top_n,
    )
    ranking_branch = "reranked" if "reranked" in branch_scores else "fused"
    ranking_scores = branch_scores[ranking_branch]
    rank_result = compute_retrieval_ranks(
        corpus,
        query_chunk_size=args.query_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
        similarities=ranking_scores,
    )
    metrics, similarities = compute_retrieval_metrics(
        corpus, query_chunk_size=args.query_chunk_size, candidate_chunk_size=args.candidate_chunk_size, rank_result=rank_result
    )
    branch_metrics = retrieval_branch_diagnostics(corpus, branch_scores)
    global_similarities = branch_scores["global"]
    local_similarities = branch_scores.get("local")
    if not torch.allclose(similarities, ranking_scores, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Selected evaluator scores disagree with independently computed branch scores")
    base_text_embeddings = corpus.teacher_text_embeddings if corpus.teacher_text_embeddings is not None else corpus.text_embeddings
    text_derived_semantic_relevance = semantic_teacher_relevance_matrix(
        base_text_embeddings,
        corpus.captions,
        corpus.caption_to_pair.long(),
        corpus.caption_group_ids.long(),
        pair_count=len(corpus.pair_ids),
        top_k=0,
    ) > 0
    pair_index = {pair_id: index for index, pair_id in enumerate(corpus.pair_ids)}
    sample_by_id = _sample_map(dataset)
    query_items = _query_items(args, dataset, pair_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals = args.output_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "qcpr_mask_overlays"
    if local_similarities is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    latent_positive_audit_path = args.output_dir / "latent_positive_audit.jsonl"
    hard_negative_audit_path = args.output_dir / "hard_negative_audit.jsonl"
    candidate_captions = [""] * len(corpus.pair_ids)
    for caption, pair in zip(corpus.captions, corpus.caption_to_pair.tolist(), strict=True):
        if not candidate_captions[int(pair)]:
            candidate_captions[int(pair)] = caption
    with latent_positive_audit_path.open("w", encoding="utf-8") as latent_handle:
        for query_index, query_caption in enumerate(corpus.captions):
            query_signature = _structured_signature(query_caption)
            ranked = torch.argsort(ranking_scores[query_index], descending=True, stable=True)[:10]
            for candidate_index in ranked.tolist():
                if text_derived_semantic_relevance[query_index, candidate_index]:
                    continue
                candidate_signature = _structured_signature(candidate_captions[candidate_index])
                failed = []
                for attribute, key in (("object", "objects"), ("direction", "direction"), ("location", "locations"), ("count", "counts"), ("relation", "relation")):
                    query_value, candidate_value = query_signature[key], candidate_signature[key]
                    if not query_value or query_value in {"unknown", "none"}:
                        continue
                    matched = bool(query_value & candidate_value) if isinstance(query_value, frozenset) else query_value == candidate_value
                    if not matched:
                        failed.append(attribute)
                latent_handle.write(json.dumps({
                    "query_index": query_index,
                    "query": query_caption,
                    "candidate_index": candidate_index,
                    "candidate_pair_id": corpus.pair_ids[candidate_index],
                    "candidate_caption": candidate_captions[candidate_index],
                    "ranking_score": float(ranking_scores[query_index, candidate_index]),
                    "semantic": False,
                    "relevance_rules_failed": failed,
                    "training_negative": False,
                }) + "\n")
    hard_masks = structured_hard_negative_masks(corpus.captions, corpus.caption_to_pair, len(corpus.pair_ids))
    with hard_negative_audit_path.open("w", encoding="utf-8") as hard_handle:
        for category, mask in hard_masks.items():
            for query_index in torch.nonzero(mask.any(dim=1), as_tuple=False).flatten().tolist():
                candidate_indices = torch.nonzero(mask[query_index], as_tuple=False).flatten()
                candidate_index = int(candidate_indices[ranking_scores[query_index, candidate_indices].argmax()].item())
                hard_handle.write(json.dumps({
                    "category": category,
                    "query_index": query_index,
                    "query": corpus.captions[query_index],
                    "candidate_index": candidate_index,
                    "candidate_pair_id": corpus.pair_ids[candidate_index],
                    "candidate_caption": candidate_captions[candidate_index],
                    "scores": {name: float(values[query_index, candidate_index]) for name, values in branch_scores.items()},
                    "excluded_as_latent_positive": bool(text_derived_semantic_relevance[query_index, candidate_index]),
                }) + "\n")
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
        relevant_mask = text_derived_semantic_relevance[caption_index]
        scores = similarities[caption_index]
        global_scores = global_similarities[caption_index]
        local_scores = local_similarities[caption_index] if local_similarities is not None else None
        order = rank_result.ranked_candidate_indices[caption_index]
        relevant_positions = torch.nonzero(relevant_mask[order], as_tuple=False).flatten()
        relevant_rank = int(relevant_positions[0].item() + 1) if relevant_positions.numel() else len(corpus.pair_ids) + 1
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
            candidate_sample = sample_by_id[retrieved_id]
            candidate_captions = candidate_sample.get("captions", []) if isinstance(candidate_sample, dict) else (candidate_sample.captions or [])
            result_record = {
                    "rank": row,
                    "pair_id": retrieved_id,
                    "query": query_caption,
                    "candidate_captions": candidate_captions,
                    "candidate_metadata": candidate_sample.get("source_metadata", {}) if isinstance(candidate_sample, dict) else getattr(candidate_sample, "metadata", {}),
                    "segmentation_target_kind": corpus.segmentation_target_kinds[int(retrieved_index)] if corpus.segmentation_target_kinds else "none",
                    "segmentation_supervision_weight": float(corpus.segmentation_weights[int(retrieved_index)]) if corpus.segmentation_weights is not None else 0.0,
                    "score": score,
                    "S_global": global_score,
                    "S_local": local_score,
                    "S_final": score,
                    "global_score": global_score,
                    "local_score": local_score,
                    "final_score": score,
                    "score_mode": corpus.score_mode,
                    "text_derived_semantic_relevance": is_relevant,
                    "exact_pair_relevance": is_exact,
                    "relevance_rules_passed": [name for name, passed in (("semantic", is_relevant), ("exact_pair", is_exact)) if passed],
                    "relevance_rules_failed": [name for name, passed in (("semantic", is_relevant), ("exact_pair", is_exact)) if not passed],
                }
            if corpus.patch_tokens is not None and (corpus.mask_query_embeddings is not None or corpus.qcpr_architecture_version == "v2"):
                candidate_patches = corpus.patch_tokens[int(retrieved_index)].float()
                patch_logits = _result_logits(corpus, caption_index, int(retrieved_index))
                if corpus.qcpr_architecture_version == "v1":
                    recomputed_local = float(patch_logits.sigmoid().amax().item())
                else:
                    pooled = QCPRPatchReranker.masked_local_embedding(candidate_patches.unsqueeze(0), patch_logits.reshape(1, 1, -1))
                    recomputed_local = float(torch.einsum("d,qbd->qb", corpus.text_embeddings[caption_index].float(), pooled)[0, 0].item())
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
                regions = _top_regions(soft, args.mask_threshold)
                evidence = _model_evidence(soft, regions)
                render_channels = {name: np.zeros(t2.shape[:2]) for name in ("changed", "appeared", "disappeared")}
                if corpus.temporal_explanation_logits is not None:
                    probabilities = corpus.temporal_explanation_logits[int(retrieved_index)].sigmoid().T
                    for channel_index, channel_name in enumerate(("changed", "appeared", "disappeared")):
                        render_channels[channel_name] = _candidate_mask(torch.logit(probabilities[channel_index].clamp(1e-6, 1 - 1e-6)), t2.shape[:2])

                panel, panel_axes = plt.subplots(3, 4, figsize=(18, 13))
                panel_axes[0, 0].imshow(t1); panel_axes[0, 0].set_title("T1")
                panel_axes[0, 1].imshow(t2); panel_axes[0, 1].set_title("T2")
                panel_axes[0, 2].imshow(rgb_absolute_difference(t1, t2), cmap="magma"); panel_axes[0, 2].set_title("RGB difference")
                panel_axes[0, 3].imshow(gt if gt is not None else np.zeros(t2.shape[:2]), cmap="gray"); panel_axes[0, 3].set_title("GT mask" if gt is not None else "GT unavailable")
                panel_axes[1, 0].imshow(t1); panel_axes[1, 0].imshow(soft, cmap="magma", alpha=.55, vmin=0, vmax=1); panel_axes[1, 0].set_title("Soft mask over T1")
                panel_axes[1, 1].imshow(t2); panel_axes[1, 1].imshow(soft, cmap="magma", alpha=.55, vmin=0, vmax=1); panel_axes[1, 1].set_title("Soft mask over T2")
                panel_axes[1, 2].imshow(thresholded, cmap="gray"); panel_axes[1, 2].set_title(f"Thresholded @{args.mask_threshold:g}")
                panel_axes[1, 3].axis("off"); panel_axes[1, 3].text(0, .95, f"pair_id: {retrieved_id}\nS_global={global_score:.4f}\nS_local={local_score:.4f}\nS_final={score:.4f}\nsemantic={is_relevant}\nexact={is_exact}", va="top")
                render_temporal_channel_panels(panel_axes[2, :3], render_channels, temporal_channel_status)
                panel_axes[2, 3].axis("off")

                for axis in panel_axes.flat: axis.axis("off")
                panel.suptitle(f"Exact query: {query_caption}")
                panel.tight_layout(rect=(0, 0, 1, .96))
                panel_path = visuals / f"{stem}.png"
                panel.savefig(panel_path, dpi=140, bbox_inches="tight"); plt.close(panel)
                result_record.update({
                    "query": query_caption, "soft_mask_path": str(soft_path),
                    "S_token_patch": float(patch_logits.sigmoid().amax().item()),
                    "thresholded_mask_path": str(threshold_path), "explanation_panel_path": str(panel_path),
                    "per_result_Dice": dice, "per_result_IoU": iou, "top_regions": regions,
                    **evidence,
                })
                if corpus.temporal_explanation_logits is not None and temporal_channel_status["temporal_channels_trained"]:
                    channels = corpus.temporal_explanation_logits[int(retrieved_index)].sigmoid().T
                    channel_paths = {}
                    for channel_index, channel_name in enumerate(("changed", "appeared", "disappeared")):
                        channel_mask = _candidate_mask(
                            torch.logit(channels[channel_index].clamp(1e-6, 1 - 1e-6)), t2.shape[:2]
                        )
                        channel_path = overlay_dir / f"{stem}_{channel_name}.png"
                        Image.fromarray(np.uint8(channel_mask * 255), mode="L").save(channel_path)
                        channel_paths[f"{channel_name}_probability_map_path"] = str(channel_path)
                        channel_paths[f"{channel_name}_mean_probability"] = float(channel_mask.mean())
                    result_record["temporal_direction_evidence"] = {name: channel_paths[f"{name}_mean_probability"] for name in ("changed", "appeared", "disappeared")}
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
        "query_chunk_size": args.query_chunk_size,
        "candidate_chunk_size": args.candidate_chunk_size,
        "rerank_top_n": args.rerank_top_n,
        "ranking_branch": ranking_branch,
        "num_queries": int(corpus.text_embeddings.shape[0]),
        "num_candidates": int(corpus.pair_embeddings.shape[0]),
        "metric_query_count": int(corpus.text_embeddings.shape[0]),
        "rendered_query_count": len(records),
        "candidate_count": int(corpus.pair_embeddings.shape[0]),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_gpu_allocated_bytes": corpus.peak_allocated_vram_bytes,
        "peak_gpu_reserved_bytes": corpus.peak_reserved_vram_bytes,
        "evaluator_wall_seconds": float(time.perf_counter() - evaluation_started),
        "evaluation_wall_seconds": float(time.perf_counter() - evaluation_started),
        "ranking_passes": 1,
        "baseline_identity": "clean_independent_pretrained_jina_universat",
        "historical_e0_continuity": False,
        "relevance_contract": {
            "text_derived_semantic_relevance": "caption-similarity pseudo-target from frozen pre-adapter Jina embeddings",
            "image_relevance_ground_truth": False,
            "historical_global_teacher_used": False,
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "stage1_next": bool(checkpoint.get("stage1_next")),
        "checkpoint_epoch": checkpoint.get("epoch", checkpoint.get("epoch_index")),
        "checkpoint_step": checkpoint.get("step"),
        "split": args.split,
        **_eval_metadata(args, dataset, data_mode),
        "corpus_metrics": metrics,
        "retrieval_branch_metrics": branch_metrics,
        "hard_negative_audit_path": str(hard_negative_audit_path),
        "latent_positive_audit_path": str(latent_positive_audit_path),
        "retrieval_branch_comparison_path": str(args.output_dir / "retrieval_branch_comparison.json"),
        "structured_retrieval_metrics_path": str(args.output_dir / "structured_retrieval_metrics.json"),
        "score_calibration_path": str(args.output_dir / "score_calibration.json"),
        "gallery_metrics": gallery_metrics,
        "top_k": args.top_k,
        "patch_reranker_available": corpus.patch_tokens is not None,
        "qcpr_score_mode": corpus.score_mode,
        "qcpr_architecture_version": corpus.qcpr_architecture_version,
        **temporal_channel_status,
        **{
            key: value
            for key, value in metrics.items()
            if key.startswith("mask_") or key in {"predicted_mask_area_mean", "target_mask_area_mean"}
        },
    }
    _assert_no_nonfinite(report)
    (args.output_dir / "retrieval_branch_comparison.json").write_text(
        json.dumps(branch_metrics, indent=2), encoding="utf-8"
    )
    structured_metrics = {
        branch: {
            key: value for key, value in values.items()
            if any(key.startswith(prefix) for prefix in ("object_match_", "direction_match_", "location_match_", "count_match_", "relation_match_"))
        }
        for branch, values in branch_metrics.items()
    }
    (args.output_dir / "structured_retrieval_metrics.json").write_text(
        json.dumps(structured_metrics, indent=2), encoding="utf-8"
    )
    calibration = {
        branch: {
            key: value for key, value in values.items()
            if key in {"ECE", "Brier"} or "score_" in key or "margin_" in key or key.startswith("top1_minus_top2_")
        }
        for branch, values in branch_metrics.items()
    }
    calibration["fusion_weights"] = (
        corpus.qcpr_reranker.fusion_logits.detach().cpu().softmax(dim=0).tolist()
        if corpus.qcpr_architecture_version == "v2" and corpus.qcpr_reranker is not None
        else [corpus.qcpr_alpha, corpus.qcpr_beta]
    )
    if corpus.qcpr_architecture_version == "v2" and corpus.qcpr_reranker is not None:
        calibration["branch_log_scales"] = corpus.qcpr_reranker.branch_log_scales.detach().cpu().tolist()
        calibration["branch_positive_scales"] = corpus.qcpr_reranker.branch_log_scales.detach().cpu().exp().tolist()
        calibration["branch_fixed_biases"] = corpus.qcpr_reranker.branch_biases.detach().cpu().tolist()
    correlations: dict[str, float] = {}
    branch_names = [name for name in ("global", "local", "token_patch", "fused", "reranked") if name in branch_scores]
    for left_index, left_name in enumerate(branch_names):
        left = branch_scores[left_name].float().flatten()
        for right_name in branch_names[left_index + 1:]:
            right = branch_scores[right_name].float().flatten()
            if left.numel() > 1 and float(left.std()) > 0.0 and float(right.std()) > 0.0:
                value = float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())
            else:
                value = 0.0
            correlations[f"{left_name}__{right_name}"] = value
    calibration["branch_score_correlations"] = correlations
    (args.output_dir / "score_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "retrieval_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_progress(
        progress_path,
        stage="complete",
        completed=1,
        total=1,
        started=evaluation_started,
        complete=True,
        metrics={"corpus_metrics": metrics, "gallery_metrics": gallery_metrics},
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
