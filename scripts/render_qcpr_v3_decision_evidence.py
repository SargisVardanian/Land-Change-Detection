#!/usr/bin/env python3
"""Inference-only counterfactual decision-evidence renderer for QCPR v3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import qcpr_v3_data_compat as data_compat
from evaluate_qcpr_v3_fast import make_config
from land_change_detection.models.qcpr_v3 import QCPRV3Config
from land_change_detection.models.qcpr_v3_explanations import (
    apply_selected_counterfactual,
    contribution_mass_selection,
    counterfactual_evidence,
    reconstruct_final_score,
    score_variants,
)
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model


def _opposite_query(query: str, direction: str) -> str:
    source, target = ("appeared", "disappeared") if direction == "appeared" else ("disappeared", "appeared")
    swapped = query.replace(source, target).replace(source.capitalize(), target.capitalize())
    if swapped == query:
        swapped = f"objects {target} in the changed region"
    return swapped


def _rgb(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((shape[1], shape[0]), Image.Resampling.BILINEAR)) / 255.0


def _upsample_grid(values: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    return F.interpolate(values[None, None].float(), shape, mode="nearest")[0, 0].numpy()


def _render_panel(path: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(len(records), 8, figsize=(25, 3.2 * len(records)), squeeze=False, constrained_layout=True)
    titles = ("T1", "T2", "RGB |T2-T1|", "signed score drop", "positive evidence", "80% causal segment", "causal overlay", "decoded mask\n(non-causal diagnostic)")
    for row, record in enumerate(records):
        t1, t2 = record.pop("_t1"), record.pop("_t2")
        signed, positive, segment, decoded = record.pop("_signed"), record.pop("_positive"), record.pop("_segment"), record.pop("_decoded")
        difference = np.abs(t2 - t1)
        overlay = t2.copy(); overlay[segment] = 0.45 * overlay[segment] + 0.55 * np.array([1.0, 0.15, 0.05])
        shown = (t1, t2, difference, signed, positive, segment, overlay, decoded)
        for column, image in enumerate(shown):
            axis = axes[row, column]
            if column == 3:
                bound = max(abs(float(signed.min())), abs(float(signed.max())), 1e-8)
                axis.imshow(image, cmap="coolwarm", vmin=-bound, vmax=bound)
            elif column in {4, 7}:
                axis.imshow(image, cmap="magma", vmin=0)
            else:
                axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(titles[column], fontsize=9); axis.axis("off")
        axes[row, 0].set_ylabel(
            f"{record['pair_id']}\n{record['direction']} | {record['status']}\n"
            f"S={record['scores']['final']:+.4f}; top drop={record['faithfulness']['top_evidence_drop']:+.4f}\n"
            f"random={record['faithfulness']['random_drop_mean']:+.4f}±{record['faithfulness']['random_drop_std']:.4f}",
            fontsize=8,
        )
    fig.suptitle("QCPR v3 inference-only counterfactual decision evidence (not a probability mask)", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--evidence-mass", type=float, default=0.8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--random-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real QCPR v3 decision-evidence rendering requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = make_config(payload, args.chunk_size, args.manifest)
    _, validation = data_compat.build_datasets(config)
    candidates = [i for i, sample in enumerate(validation.samples) if str(sample.get("dataset_name", "")).casefold() == "s2looking"]
    if args.pair_id:
        requested = set(args.pair_id)
        candidates = [i for i in candidates if str(validation.samples[i].get("pair_id")) in requested]
    candidates = candidates[: args.max_pairs]
    if not candidates:
        raise RuntimeError("no requested S2Looking rows were found")
    model = build_clean_v3_model(
        QCPRV3BackboneConfig(**payload["backbone_config"]), device=device,
        grounding_config=QCPRV3Config(**payload["grounding_config"]),
    )
    model.load_state_dict(payload["model"], strict=True); model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    for index in candidates:
        sample = validation.samples[index]
        batch = next(iter(data_compat.make_eval_loader(Subset(validation, [index]), config)))
        images = batch["images"][0]
        temporal = batch["temporal_valid_mask"][0]
        query = str(batch["captions"][0])
        direction = "disappeared" if "disappear" in query.casefold() else "appeared"
        scorer = lambda values: score_variants(model, values, query, temporal, chunk_size=args.chunk_size)
        original, signed, boxes = counterfactual_evidence(scorer, images, grid=args.grid, direction=direction, chunk_size=args.chunk_size)
        positive = signed.clamp_min(0)
        selection = contribution_mass_selection(positive, mass=args.evidence_mass)
        selected_variant = apply_selected_counterfactual(images, boxes, selection.mask, direction=direction)
        selected_score = scorer(selected_variant.unsqueeze(0))[0]
        top_drop = float(original - selected_score)
        random_drops = []
        for _ in range(args.random_trials):
            random_mask = torch.zeros(args.grid * args.grid, dtype=torch.bool)
            if selection.selected_count:
                random_mask[torch.randperm(random_mask.numel(), generator=generator)[: selection.selected_count]] = True
            random_variant = apply_selected_counterfactual(images, boxes, random_mask.reshape(args.grid, args.grid), direction=direction)
            random_drops.append(float(original - scorer(random_variant.unsqueeze(0))[0]))
        original_output = model(images.unsqueeze(0).to(device), [query], temporal_valid_mask=temporal[None].to(device))
        reconstruct_final_score(original_output.scores, model.grounder.rerank_logits)
        weights = F.softplus(model.grounder.rerank_logits)
        opposite = _opposite_query(query, direction)
        opposite_score = float(score_variants(model, images.unsqueeze(0), opposite, temporal, chunk_size=1)[0])
        height, width = images.shape[-2:]
        signed_map = _upsample_grid(signed, (height, width)); positive_map = _upsample_grid(positive, (height, width))
        segment = _upsample_grid(selection.mask.float(), (height, width)).astype(bool)
        decoded = original_output.scores.decoded_mask_logits[0, 0].sigmoid().float().cpu()
        decoded = F.interpolate(decoded[None, None], (height, width), mode="bilinear", align_corners=False)[0, 0].numpy()
        random_array = np.asarray(random_drops)
        score = original_output.scores
        record = {
            "pair_id": str(sample.get("pair_id")), "query": query, "opposite_query": opposite, "direction": direction,
            "status": selection.status,
            "scores": {
                "global": float(score.global_score[0, 0]),
                "weighted_local": float(weights[0] * score.local_score[0, 0]),
                "weighted_token_patch": float(weights[1] * score.token_patch_score[0, 0]),
                "final": float(score.reranked_score[0, 0]), "opposite_query_final": opposite_score,
            },
            "evidence": {"grid": args.grid, "positive_mass": selection.positive_mass, "selected_tiles": selection.selected_count, "mass_fraction": args.evidence_mass, "signed_drops": signed.tolist()},
            "faithfulness": {
                "top_evidence_drop": top_drop, "random_drop_mean": float(random_array.mean()), "random_drop_std": float(random_array.std()),
                "top_drop_positive": top_drop > 0, "top_beats_random": top_drop > float(random_array.mean()),
                "no_full_frame_fallback": not (selection.status == "NO_FAITHFUL_SPATIAL_EVIDENCE" and bool(selection.mask.any())),
            },
            "_t1": _rgb(sample["t1_path"], (height, width)), "_t2": _rgb(sample["t2_path"], (height, width)),
            "_signed": signed_map, "_positive": positive_map, "_segment": segment, "_decoded": decoded,
        }
        records.append(record)
    serializable = [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]
    report = {
        "schema_version": "qcpr-v3-counterfactual-decision-evidence-v1", "checkpoint": str(args.checkpoint),
        "checkpoint_commit": payload.get("commit"), "inference_only": True, "optimizer_steps": 0,
        "evidence_semantics": "signed S_final(original)-S_final(direction-aware tile counterfactual); not probability",
        "decoded_mask_semantics": "separate non-causal diagnostic only", "records": serializable,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "decision_evidence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _render_panel(args.output_dir / "decision_evidence_panel.png", records)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
