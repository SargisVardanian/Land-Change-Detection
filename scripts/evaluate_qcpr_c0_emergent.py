#!/usr/bin/env python3
"""One-shot post-training C0 evaluator on predeclared S2Looking development."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import qcpr_v3_data_compat as data_compat
from land_change_detection.models.qcpr_v3 import QCPRV3Config
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model
from land_change_detection.models.qcpr_v3_losses import query_mask_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def soft_iou(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).flatten(1).sum(1)
    union = (probability + target - probability * target).flatten(1).sum(1)
    return intersection / union.clamp_min(1e-6)


def save_panel(path: Path, before: torch.Tensor, after: torch.Tensor, target: torch.Tensor, probability: torch.Tensor, opposite: torch.Tensor, title: str) -> None:
    import matplotlib.pyplot as plt

    def rgb(image: torch.Tensor):
        image = image.detach().float().cpu()[:3]
        image = image - image.amin(dim=(1, 2), keepdim=True)
        image = image / image.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return image.permute(1, 2, 0).numpy()

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    for axis, data, name in zip(
        axes.flat,
        (rgb(before), rgb(after), target.cpu(), probability.cpu(), opposite.cpu(), (probability - opposite).cpu()),
        ("T1 before", "T2 after", "Directional target (evaluation only)", "C0 correct query", "C0 opposite query", "Correct - opposite"),
        strict=True,
    ):
        axis.imshow(data, cmap=None if name.startswith("T") else "magma")
        axis.set_title(name)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "examples").mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError("C0 evaluator requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    selected_report = args.checkpoint.parent / "best_feasible_development.json"
    if not selected_report.exists():
        raise RuntimeError(f"accepted B selection report is missing: {selected_report}")
    selected_payload = json.loads(selected_report.read_text())
    selected_sha = sha256(args.checkpoint)
    if not bool(selected_payload.get("feasible")):
        raise RuntimeError("B selection report is not feasible")
    if selected_payload.get("checkpoint_sha256") != selected_sha:
        raise RuntimeError(
            "B checkpoint SHA does not match accepted selection report: "
            f"{selected_sha} != {selected_payload.get('checkpoint_sha256')}"
        )
    summary_path = args.checkpoint.parent / "summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("status") != "PASS":
        raise RuntimeError("B long run is not marked PASS")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(payload["data_config"])
    config_values.update(
        batch_size=args.batch_size, num_workers=4, dataset_config=None,
        train_manifests=(str(args.manifest),), val_manifests=(str(args.manifest),),
        dataset_sampling_weights=(), target_aware_mask_crop=False,
        load_segmentation_targets=True,
    )
    fields = data_compat.Stage1NextConfig.__dataclass_fields__
    config = data_compat.Stage1NextConfig(**{key: value for key, value in config_values.items() if key in fields})
    _, dataset = data_compat.build_datasets(config)
    model = build_clean_v3_model(
        QCPRV3BackboneConfig(**payload["backbone_config"]), device=device,
        grounding_config=QCPRV3Config(**payload["grounding_config"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    records: list[dict[str, object]] = []
    all_logits, all_targets = [], []
    loader = data_compat.make_eval_loader(dataset, config)
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            temporal = batch["temporal_valid_mask"].to(device)
            mapping = batch["caption_to_pair"].to(device)
            pair, patches, _ = model.encode_pairs(images, temporal)
            _, text, tokens, attention, content = model.encode_texts(batch["captions"])
            output = model.score_encoded(
                pair, patches, text.to(device), tokens.to(device), attention.to(device),
                content.to(device), decode_mask=False,
            )
            rows = torch.arange(mapping.numel(), device=device)
            probability = output.emergent_soft_map[rows, mapping]
            targets = batch["query_masks"].to(device).float()
            targets = F.interpolate(targets[:, None], probability.shape[-2:], mode="nearest")[:, 0]
            logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            pair_rows: dict[int, list[int]] = {}
            for query_index, pair_index in enumerate(mapping.tolist()):
                pair_rows.setdefault(int(pair_index), []).append(query_index)
            opposite = probability.clone()
            for indices in pair_rows.values():
                if len(indices) == 2:
                    opposite[indices[0]] = probability[indices[1]]
                    opposite[indices[1]] = probability[indices[0]]
            correct_iou = soft_iou(probability, targets)
            opposite_iou = soft_iou(opposite, targets)
            for index, caption in enumerate(batch["captions"]):
                item_metrics = query_mask_metrics(logits[index:index + 1].cpu(), targets[index:index + 1].cpu())
                record = {
                    "pair_id": str(batch["pair_ids"][int(mapping[index])]),
                    "query": str(caption), "direction": "disappeared" if "disappear" in caption.lower() or "demol" in caption.lower() else "appeared",
                    "correct_soft_iou": float(correct_iou[index]), "opposite_soft_iou": float(opposite_iou[index]),
                    "soft_query_swap_gap": float(correct_iou[index] - opposite_iou[index]), **item_metrics,
                }
                records.append(record)
                if len(records) <= 16:
                    save_panel(
                        args.output_dir / "examples" / f"example_{len(records):03d}.png",
                        images[int(mapping[index]), 0], images[int(mapping[index]), 1], targets[index],
                        probability[index], opposite[index], f"{record['pair_id']} | {caption}",
                    )
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    summary = query_mask_metrics(logits, targets)
    for direction in ("appeared", "disappeared"):
        subset = [row for row in records if row["direction"] == direction]
        summary[f"{direction}_query_swap_gap"] = sum(float(row["soft_query_swap_gap"]) for row in subset) / max(len(subset), 1)
    summary.update(
        status="POST_TRAINING_DEVELOPMENT_ONLY", checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint), manifest=str(args.manifest), manifest_sha256=sha256(args.manifest),
        accepted_selection_report=str(selected_report), accepted_selection_sha256=selected_sha,
        mask_pixels_used_for_training=False, supervised_mask_decoder_used=False,
        scientific_claim="weakly supervised emergent soft localization",
    )
    (args.output_dir / "c0_emergent_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "qualitative_examples.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
