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
    fig, axes = plt.subplots(len(records), 9, figsize=(28, 3.2 * len(records)), squeeze=False, constrained_layout=True)
    titles = ("T1", "T2", "RGB |T2-T1|", "raw signed drop", "color-matched\nsigned drop", "80% contribution\nregion", "decision-sensitivity\noverlay", "opposite-query\nsigned drop", "decoded mask\n(non-causal diagnostic)")
    for row, record in enumerate(records):
        t1, t2 = record.pop("_t1"), record.pop("_t2")
        raw_signed, signed, segment, opposite_signed, decoded = (record.pop("_raw_signed"), record.pop("_signed"), record.pop("_segment"), record.pop("_opposite_signed"), record.pop("_decoded"))
        difference = np.abs(t2 - t1)
        overlay = t2.copy(); overlay[segment] = 0.45 * overlay[segment] + 0.55 * np.array([1.0, 0.15, 0.05])
        shown = (t1, t2, difference, raw_signed, signed, segment, overlay, opposite_signed, decoded)
        for column, image in enumerate(shown):
            axis = axes[row, column]
            if column in {3, 4, 7}:
                bound = max(abs(float(image.min())), abs(float(image.max())), 1e-8)
                axis.imshow(image, cmap="coolwarm", vmin=-bound, vmax=bound)
            elif column == 8:
                axis.imshow(image, cmap="magma", vmin=0)
            else:
                axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(titles[column], fontsize=9); axis.axis("off")
        axes[row, 0].set_ylabel(
            f"{record['pair_id']}\n{record['direction']} | {record['status']}\n"
            f"S={record['scores']['final']:+.4f}; matched top drop={record['faithfulness']['color_matched']['top_evidence_drop']:+.4f}\n"
            f"random+2σ={record['faithfulness']['color_matched']['random_gate']:+.4f}",
            fontsize=8,
        )
    fig.suptitle("QCPR v3 counterfactual decision-sensitivity regions; no column is a probability explanation", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", action="append", required=True, help="Explicit fixed pair ID; repeat 2–6 times.")
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
    if not 2 <= len(args.pair_id) <= 6:
        raise ValueError("provide 2–6 explicit --pair-id values")
    by_id = {str(sample.get("pair_id")): index for index, sample in enumerate(validation.samples)}
    missing = [pair_id for pair_id in args.pair_id if pair_id not in by_id]
    if missing:
        raise RuntimeError(f"requested pair IDs not present in manifest: {missing}")
    candidates = [by_id[pair_id] for pair_id in args.pair_id]
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
        mode_results = {}
        for replacement in ("raw", "color_matched"):
            original, signed, boxes = counterfactual_evidence(scorer, images, grid=args.grid, direction=direction, chunk_size=args.chunk_size, replacement=replacement)
            selection = contribution_mass_selection(signed.clamp_min(0), mass=args.evidence_mass)
            selected_variant = apply_selected_counterfactual(images, boxes, selection.mask, direction=direction, replacement=replacement)
            top_drop = float(original - scorer(selected_variant.unsqueeze(0))[0])
            random_drops = []
            for _ in range(args.random_trials):
                random_mask = torch.zeros(args.grid * args.grid, dtype=torch.bool)
                if selection.selected_count:
                    random_mask[torch.randperm(random_mask.numel(), generator=generator)[: selection.selected_count]] = True
                random_variant = apply_selected_counterfactual(images, boxes, random_mask.reshape(args.grid, args.grid), direction=direction, replacement=replacement)
                random_drops.append(float(original - scorer(random_variant.unsqueeze(0))[0]))
            random_array = np.asarray(random_drops)
            random_gate = float(random_array.mean() + 2.0 * random_array.std())
            mode_results[replacement] = {"signed": signed, "selection": selection, "top_drop": top_drop, "random": random_array, "random_gate": random_gate, "passes": selection.selected_count > 0 and top_drop > 0 and top_drop > random_gate}
        original_output = model(images.unsqueeze(0).to(device), [query], temporal_valid_mask=temporal[None].to(device))
        reconstruct_final_score(original_output.scores, model.grounder.rerank_logits)
        weights = F.softplus(model.grounder.rerank_logits)
        opposite = _opposite_query(query, direction)
        opposite_scorer = lambda values: score_variants(model, values, opposite, temporal, chunk_size=args.chunk_size)
        opposite_score, opposite_signed, _ = counterfactual_evidence(opposite_scorer, images, grid=args.grid, direction=direction, chunk_size=args.chunk_size, replacement="color_matched")
        height, width = images.shape[-2:]
        raw = mode_results["raw"]; matched = mode_results["color_matched"]
        signed_map = _upsample_grid(matched["signed"], (height, width))
        raw_signed_map = _upsample_grid(raw["signed"], (height, width))
        opposite_signed_map = _upsample_grid(opposite_signed, (height, width))
        segment = _upsample_grid(matched["selection"].mask.float(), (height, width)).astype(bool)
        decoded = original_output.scores.decoded_mask_logits[0, 0].sigmoid().float().cpu()
        decoded = F.interpolate(decoded[None, None], (height, width), mode="bilinear", align_corners=False)[0, 0].numpy()
        score = original_output.scores
        status = "FAITHFUL_DECISION_SENSITIVITY" if raw["passes"] and matched["passes"] else "INCONCLUSIVE_DECISION_SENSITIVITY"
        record = {
            "pair_id": str(sample.get("pair_id")), "query": query, "opposite_query": opposite, "direction": direction,
            "status": status,
            "scores": {
                "global": float(score.global_score[0, 0]),
                "weighted_local": float(weights[0] * score.local_score[0, 0]),
                "weighted_token_patch": float(weights[1] * score.token_patch_score[0, 0]),
                "final": float(score.reranked_score[0, 0]), "opposite_query_final": float(opposite_score),
            },
            "evidence": {"grid": args.grid, "mass_fraction": args.evidence_mass, "raw_signed_drops": raw["signed"].tolist(), "color_matched_signed_drops": matched["signed"].tolist(), "opposite_query_color_matched_signed_drops": opposite_signed.tolist()},
            "faithfulness": {name: {"status": value["selection"].status, "positive_mass": value["selection"].positive_mass, "selected_tiles": value["selection"].selected_count, "top_evidence_drop": value["top_drop"], "random_drop_mean": float(value["random"].mean()), "random_drop_std": float(value["random"].std()), "random_gate": value["random_gate"], "passes_random_plus_2std": value["passes"], "no_full_frame_fallback": not (value["selection"].status == "NO_FAITHFUL_SPATIAL_EVIDENCE" and bool(value["selection"].mask.any()))} for name, value in mode_results.items()},
            "_t1": _rgb(sample["t1_path"], (height, width)), "_t2": _rgb(sample["t2_path"], (height, width)),
            "_raw_signed": raw_signed_map, "_signed": signed_map, "_segment": segment, "_opposite_signed": opposite_signed_map, "_decoded": decoded,
        }
        records.append(record)
    serializable = [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]
    report = {
        "schema_version": "qcpr-v3-counterfactual-decision-evidence-v1", "checkpoint": str(args.checkpoint),
        "checkpoint_commit": payload.get("commit"), "inference_only": True, "optimizer_steps": 0,
        "evidence_semantics": "signed S_final(original)-S_final(direction-aware tile counterfactual); decision sensitivity, not probability or causal proof",
        "decoded_mask_semantics": "separate non-causal diagnostic only", "records": serializable,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "decision_evidence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _render_panel(args.output_dir / "decision_evidence_panel.png", records)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
