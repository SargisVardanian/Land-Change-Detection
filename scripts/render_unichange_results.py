#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.backbones.universat_backend import UniverSatBackendConfig, UniverSatJointBackend
from land_change_detection.data.unichange_mci import UniChangeMciDataset, collate_unichange_mci
from land_change_detection.data.unichange_subset import resolve_levir_mci_root
from land_change_detection.models.unichange_model import UniChangeConfig, UniChangeModel


def _image(ax: plt.Axes, tensor: torch.Tensor, title: str) -> None:
    ax.imshow(tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax.set_title(title)
    ax.axis("off")


def _mask(ax: plt.Axes, tensor: torch.Tensor, title: str) -> None:
    ax.imshow(tensor.detach().cpu().float().numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render UniChange inference panels for the fixed MCI subset.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--universat-source", type=Path, required=True)
    parser.add_argument("--universat-checkpoint", type=Path, required=True)
    parser.add_argument("--jina-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_levir_mci_root(args.data_root).root
    dataset = UniChangeMciDataset(root, split="train", image_size=224, subset_file=args.subset_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_unichange_mci)
    model = UniChangeModel(
        UniverSatJointBackend(UniverSatBackendConfig(source_dir=args.universat_source, checkpoint_dir=args.universat_checkpoint)),
        JinaV5TextEncoder(JinaV5TextConfig(model_path=args.jina_model, max_length=64, freeze=True)),
        UniChangeConfig(event_queries=16),
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint.get("heads", checkpoint.get("model", {})), strict=False)
    model.eval()
    panel_dir = args.output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.limit:
                break
            out = model(batch["t1"].to(args.device), batch["t2"].to(args.device), batch["captions"], temporal_context=batch["temporal_context"])
            presence = torch.sigmoid(out.event_presence_logits[0]).detach().cpu()  # type: ignore[index]
            event_masks = out.event_masks[0].detach().cpu().view(-1, 36, 36)  # type: ignore[index]
            union = 1.0 - torch.prod(1.0 - event_masks * (presence >= 0.5).float().view(-1, 1, 1), dim=0)
            text_mask = out.text_conditioned_mask[0, 0].detach().cpu().view(36, 36) if out.text_conditioned_mask is not None and out.text_conditioned_mask.ndim == 3 else union
            gt = batch["masks"][0]
            gt_36 = F.interpolate(gt[None, None], size=(36, 36), mode="nearest").squeeze()
            diff = (batch["t2"][0] - batch["t1"][0]).abs()
            overlay = batch["t2"][0].clone()
            up_union = F.interpolate(union[None, None], size=overlay.shape[-2:], mode="bilinear", align_corners=False).squeeze()
            overlay[0] = torch.clamp(overlay[0] * 0.55 + up_union * 0.45, 0, 1)
            top_events = torch.topk(presence, k=min(4, presence.numel())).indices
            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            _image(axes[0, 0], batch["t1"][0], "T1")
            _image(axes[0, 1], batch["t2"][0], "T2")
            _image(axes[0, 2], diff, "RGB abs diff")
            _mask(axes[0, 3], gt_36, "GT mask")
            _mask(axes[1, 0], union, "Predicted union")
            _mask(axes[1, 1], text_mask, "Text-conditioned")
            _image(axes[1, 2], overlay, "Overlay on T2")
            axes[1, 3].bar(range(len(presence)), presence.numpy())
            axes[1, 3].set_title("Event presence")
            for slot, event_index in enumerate(top_events.tolist()):
                _mask(axes[2, slot], event_masks[event_index], f"Event {event_index}")
            for slot in range(len(top_events), 4):
                axes[2, slot].axis("off")
            caption = batch["captions"][0]
            fig.suptitle(f"{batch['pair_ids'][0]} | {caption[:160]}")
            fig.tight_layout()
            panel_path = panel_dir / f"{batch['pair_ids'][0]}.png"
            fig.savefig(panel_path)
            plt.close(fig)
            sidecar = {
                "pair_id": batch["pair_ids"][0],
                "caption": caption,
                "event_presence": presence.tolist(),
                "top_event_indices": top_events.tolist(),
                "subset_file": str(args.subset_file),
            }
            panel_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
            entries.append(f'<figure><img src="panels/{panel_path.name}"><figcaption>{batch["pair_ids"][0]}</figcaption></figure>')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.html").write_text("<html><body><h1>UniChange Results</h1>" + "\n".join(entries) + "</body></html>")


if __name__ == "__main__":
    main()
