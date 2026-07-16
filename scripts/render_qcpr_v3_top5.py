#!/usr/bin/env python3
"""Render an honest, labelled Top-5 QCPR v3 retrieval/grounding contact sheet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import qcpr_v3_data_compat as data_compat
from evaluate_qcpr_v3_fast import collect, make_config
from land_change_detection.models.qcpr_v3 import QCPRV3Config
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model


def _rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _mask(path: str | Path | None, size: tuple[int, int]) -> np.ndarray:
    if not path:
        return np.zeros(size, dtype=np.float32)
    image = Image.open(path).convert("L").resize((size[1], size[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def _contrast(probability: np.ndarray) -> np.ndarray:
    low, high = float(probability.min()), float(probability.max())
    return (probability - low) / max(high - low, 1e-6)


def _overlay(rgb: np.ndarray, probability: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32) / 255.0
    relative = _contrast(probability)
    heat = plt.get_cmap("magma")(relative)[..., :3]
    alpha = (0.10 + 0.60 * relative)[..., None]
    return np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0)


def render_sheet(
    output: Path,
    *,
    query: str,
    rows: list[dict[str, object]],
    checkpoint: Path,
    pool_size: int,
    top_n: int,
    rerank_weights: tuple[float, float],
    true_global_rank: int,
) -> None:
    figure = plt.figure(figsize=(22, 3.45 * len(rows) + 2.8), constrained_layout=True)
    grid = figure.add_gridspec(len(rows), 7, width_ratios=(1.35, 1, 1, 1, 1, 1, 1))
    for row_index, row in enumerate(rows):
        t1, t2 = _rgb(row["t1_path"]), _rgb(row["t2_path"])
        probability = np.asarray(row["probability"], dtype=np.float32)
        gt = _mask(row.get("mask_path"), probability.shape)
        relative = _contrast(probability)
        images = (t1, t2, gt, probability, relative, _overlay(t2, probability))
        titles = (
            "T1 (before)", "T2 (after)",
            "Generic change mask\n(context only)",
            "QUERY SOFT MASK\nabsolute probability 0..1",
            "QUERY SOFT MASK\ncontrast-enhanced diagnostic",
            "T2 + contrast-enhanced\nquery-mask overlay",
        )
        info = figure.add_subplot(grid[row_index, 0]); info.axis("off")
        role = "GROUND-TRUTH POSITIVE" if row["role"] == "ground_truth" else f"MODEL TOP-{row['final_rank']}"
        color = "#137333" if row["is_true_pair"] else "#9c2f2f"
        info.text(0.0, 0.98, role, va="top", fontsize=12, fontweight="bold", color=color)
        info.text(
            0.0, 0.84,
            f"exact match: {'YES' if row['is_true_pair'] else 'NO'}\n"
            f"pair: {row['pair_id']}\n"
            f"Stage-1 global rank: {row['global_rank']} / {pool_size}\n"
            f"selected in global Top-{top_n}: {'YES' if row['global_rank'] <= top_n else 'NO'}\n\n"
            f"S_global:  {row['global_score']:+.4f}\n"
            f"S_token:   {row['token_patch_score']:+.4f}\n"
            f"S_local:   {row['local_score']:+.4f}\n"
            f"S_rerank:  {row['reranked_score']:+.4f}\n\n"
            f"mask min/mean/max:\n{row['mask_min']:.3f} / {row['mask_mean']:.3f} / {row['mask_max']:.3f}\n"
            f"mask status: {row['mask_status']}",
            va="top", fontsize=9.2, family="monospace",
        )
        for column, (image, title) in enumerate(zip(images, titles, strict=True), start=1):
            axis = figure.add_subplot(grid[row_index, column])
            axis.imshow(
                image,
                cmap="gray" if title.startswith("Generic") else "magma" if image.ndim == 2 else None,
                vmin=0 if image.ndim == 2 else None,
                vmax=1 if image.ndim == 2 else None,
            )
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.suptitle(
        "QCPR v3 — ground-truth positive versus actual model Top-5\n"
        f"Query: “{query}”\n"
        "Exact retrieval GT = the image pair paired with this human caption. Query-specific spatial GT is NOT available; "
        "the generic change mask is shown only as context.\n"
        f"Stage 1: S_global over {pool_size} candidates → Top-{top_n}.  Stage 2: S_rerank = S_global + "
        f"{rerank_weights[0]:.3f}·S_local + {rerank_weights[1]:.3f}·S_token.  True-pair global rank={true_global_rank}.\n"
        f"Checkpoint={checkpoint.name}. IMPORTANT: smoke checkpoint (2 global + 1 mask step), not quality validated.",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--query-contains", default="two houses are built at the top")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real QCPR v3 rendering requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = make_config(payload, args.batch_size, args.manifest)
    _, validation = data_compat.build_datasets(config)
    indices = [i for i, row in enumerate(validation.samples) if str(row["dataset_name"]) in {"levir_mci", "second_cc"}][: args.pool_size]
    subset = Subset(validation, indices)
    backbone = QCPRV3BackboneConfig(**payload["backbone_config"])
    grounding = QCPRV3Config(**payload["grounding_config"])
    model = build_clean_v3_model(backbone, device=device, grounding_config=grounding)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    corpus = collect(model, subset, config, device)
    needle = args.query_contains.casefold()
    query_index = next((i for i, caption in enumerate(corpus["captions"]) if needle in caption.casefold()), 0)
    query = corpus["captions"][query_index]
    global_scores = corpus["text"][query_index] @ corpus["pairs"].T
    top_n = min(args.global_top_n, global_scores.numel())
    global_order = global_scores.argsort(descending=True)
    candidates = global_order[:top_n]
    true_pair = int(corpus["mapping"][query_index])
    true_global_rank = int(torch.nonzero(global_order == true_pair, as_tuple=False)[0]) + 1
    scoring_candidates = torch.unique(torch.cat((candidates, torch.tensor([true_pair]))), sorted=True)
    scores = model.score_encoded(
        corpus["pairs"][scoring_candidates].to(device), corpus["per_time"][scoring_candidates].to(device),
        corpus["text"][query_index:query_index + 1].to(device),
        corpus["tokens"][query_index:query_index + 1].to(device),
        corpus["attention"][query_index:query_index + 1].to(device),
        corpus["content"][query_index:query_index + 1].to(device),
    )
    top_positions = torch.searchsorted(scoring_candidates, candidates)
    order = scores.reranked_score[0, top_positions].topk(min(5, top_n)).indices.cpu()
    displayed_candidates = candidates[order]
    displayed_positions = torch.searchsorted(scoring_candidates, displayed_candidates)
    rows: list[dict[str, object]] = []
    def make_row(candidate: int, position: int, *, role: str, final_rank: int | None) -> dict[str, object]:
        sample = validation.samples[indices[candidate]]
        probability = scores.decoded_mask_logits[0, position].sigmoid().float().cpu().numpy()
        maximum = float(probability.max())
        return {
            "role": role, "final_rank": final_rank,
            "candidate_index": candidate,
            "pair_id": str(sample["pair_id"]),
            "dataset": str(sample["dataset_name"]),
            "is_true_pair": candidate == true_pair,
            "t1_path": str(sample["t1_path"]), "t2_path": str(sample["t2_path"]),
            "mask_path": str(sample.get("mask_path") or "") or None,
            "global_rank": int(torch.nonzero(global_order == candidate, as_tuple=False)[0]) + 1,
            "global_score": float(scores.global_score[0, position].cpu()),
            "token_patch_score": float(scores.token_patch_score[0, position].cpu()),
            "local_score": float(scores.local_score[0, position].cpu()),
            "reranked_score": float(scores.reranked_score[0, position].cpu()),
            "probability": probability,
            "mask_min": float(probability.min()), "mask_mean": float(probability.mean()), "mask_max": maximum,
            "mask_status": "COLLAPSED (<0.10 max)" if maximum < 0.10 else "non-collapsed",
        }
    true_position = int(torch.searchsorted(scoring_candidates, torch.tensor(true_pair)))
    rows.append(make_row(true_pair, true_position, role="ground_truth", final_rank=None))
    for final_rank, (candidate, position) in enumerate(zip(displayed_candidates.tolist(), displayed_positions.tolist(), strict=True), start=1):
        rows.append(make_row(int(candidate), int(position), role="model_top5", final_rank=final_rank))
    weights = torch.nn.functional.softplus(model.grounder.rerank_logits).detach().cpu().tolist()
    render_sheet(
        args.output, query=query, rows=rows, checkpoint=args.checkpoint, pool_size=len(indices), top_n=top_n,
        rerank_weights=(float(weights[0]), float(weights[1])), true_global_rank=true_global_rank,
    )
    serializable = [{key: value for key, value in row.items() if key != "probability"} | {
        "mask_probability_mean": float(np.asarray(row["probability"]).mean()),
        "mask_probability_min": float(np.asarray(row["probability"]).min()),
        "mask_probability_max": float(np.asarray(row["probability"]).max()),
    } for row in rows]
    report = {
        "schema_version": "qcpr-v3-top5-grounding-v1", "query": query,
        "query_index": query_index, "true_pair_candidate_index": true_pair,
        "true_pair_global_rank": true_global_rank,
        "pool_size": len(indices), "global_top_n": top_n,
        "rerank_weights": {"local": float(weights[0]), "token_patch": float(weights[1])},
        "ground_truth_contract": {
            "retrieval": "exact image pair paired with the human caption",
            "query_specific_spatial_mask": "NOT_AVAILABLE",
            "displayed_mask": "generic dataset change mask shown as non-query-specific context only",
        },
        "checkpoint": str(args.checkpoint), "checkpoint_phase": payload.get("phase"),
        "scientific_status": "SMOKE_DIAGNOSTIC_NOT_QUALITY_VALIDATED",
        "ground_truth_positive": serializable[0], "top5": serializable[1:],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
