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


def _overlay(rgb: np.ndarray, probability: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32) / 255.0
    heat = plt.get_cmap("magma")(probability)[..., :3]
    alpha = (0.15 + 0.55 * probability)[..., None]
    return np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0)


def render_sheet(
    output: Path,
    *,
    query: str,
    rows: list[dict[str, object]],
    checkpoint: Path,
    pool_size: int,
    top_n: int,
) -> None:
    figure, axes = plt.subplots(len(rows), 5, figsize=(18, 3.65 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        t1, t2 = _rgb(row["t1_path"]), _rgb(row["t2_path"])
        probability = np.asarray(row["probability"], dtype=np.float32)
        gt = _mask(row.get("mask_path"), probability.shape)
        images = (t1, t2, gt, probability, _overlay(t2, probability))
        titles = (
            "T1 (before)", "T2 (after)",
            "Generic change GT\n(not query-specific)",
            "Predicted query soft mask\nprobability 0..1", "T2 + query-mask overlay",
        )
        for axis, image, title in zip(axes[row_index], images, titles, strict=True):
            axis.imshow(image, cmap="gray" if image.ndim == 2 and title.startswith("Generic") else "magma" if title.startswith("Predicted") else None, vmin=0 if image.ndim == 2 else None, vmax=1 if image.ndim == 2 else None)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        marker = "TRUE PAIR" if row["is_true_pair"] else "candidate"
        axes[row_index, 0].set_ylabel(
            f"Rank {row_index + 1} · {marker}\n{row['pair_id']}\n"
            f"global={row['global_score']:.4f}  token={row['token_patch_score']:.4f}\n"
            f"local={row['local_score']:.4f}  reranked={row['reranked_score']:.4f}",
            fontsize=9,
        )
    figure.suptitle(
        "QCPR v3 — actual Top-5 retrieval and query-conditioned soft grounding\n"
        f"Query: “{query}”\n"
        f"Pool={pool_size}; global Top-{top_n} → canonical reranker; checkpoint={checkpoint.name}\n"
        "IMPORTANT: smoke checkpoint (2 global + 1 mask step); visualization is diagnostic, not quality validation.",
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
    candidates = global_scores.topk(top_n).indices
    scores = model.score_encoded(
        corpus["pairs"][candidates].to(device), corpus["per_time"][candidates].to(device),
        corpus["text"][query_index:query_index + 1].to(device),
        corpus["tokens"][query_index:query_index + 1].to(device),
        corpus["attention"][query_index:query_index + 1].to(device),
        corpus["content"][query_index:query_index + 1].to(device),
    )
    order = scores.reranked_score[0].topk(min(5, top_n)).indices.cpu()
    true_pair = int(corpus["mapping"][query_index])
    rows: list[dict[str, object]] = []
    for position in order.tolist():
        candidate = int(candidates[position])
        sample = validation.samples[indices[candidate]]
        rows.append({
            "candidate_index": candidate,
            "pair_id": str(sample["pair_id"]),
            "dataset": str(sample["dataset_name"]),
            "is_true_pair": candidate == true_pair,
            "t1_path": str(sample["t1_path"]), "t2_path": str(sample["t2_path"]),
            "mask_path": str(sample.get("mask_path") or "") or None,
            "global_score": float(scores.global_score[0, position].cpu()),
            "token_patch_score": float(scores.token_patch_score[0, position].cpu()),
            "local_score": float(scores.local_score[0, position].cpu()),
            "reranked_score": float(scores.reranked_score[0, position].cpu()),
            "probability": scores.decoded_mask_logits[0, position].sigmoid().float().cpu().numpy(),
        })
    render_sheet(args.output, query=query, rows=rows, checkpoint=args.checkpoint, pool_size=len(indices), top_n=top_n)
    serializable = [{key: value for key, value in row.items() if key != "probability"} | {
        "mask_probability_mean": float(np.asarray(row["probability"]).mean()),
        "mask_probability_min": float(np.asarray(row["probability"]).min()),
        "mask_probability_max": float(np.asarray(row["probability"]).max()),
    } for row in rows]
    report = {
        "schema_version": "qcpr-v3-top5-grounding-v1", "query": query,
        "query_index": query_index, "true_pair_candidate_index": true_pair,
        "pool_size": len(indices), "global_top_n": top_n,
        "checkpoint": str(args.checkpoint), "checkpoint_phase": payload.get("phase"),
        "scientific_status": "SMOKE_DIAGNOSTIC_NOT_QUALITY_VALIDATED", "top5": serializable,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
