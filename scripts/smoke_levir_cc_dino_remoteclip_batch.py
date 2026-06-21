from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from scripts.train_dino_pair_retrieval import build_dataloader, choose_device, load_retrieval_samples, set_seed
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.losses.retrieval_losses import symmetric_infonce_loss
from land_change_detection.run_metadata import path_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real LEVIR-CC DINOv2 + RemoteCLIP forward/backward batch.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dinov2-model-path", type=Path, required=True)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--remoteclip-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-feature-mode", choices=("t2_only", "signed_delta", "change_fusion"), default="signed_delta")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    samples = load_retrieval_samples(args.manifest, [], args.project_root)
    caption_samples = [sample for sample in samples if sample.caption]
    if len({sample.pair_id for sample in caption_samples}) < 2:
        raise SystemExit("Need at least two unique LEVIR-CC pair_ids for a meaningful forward/backward smoke batch.")
    args.max_train_samples = min(args.batch_size, len({sample.pair_id for sample in caption_samples}))
    args.num_workers = 0
    loader = build_dataloader(caption_samples, args, shuffle=False)
    batch = next(iter(loader))
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="dinov2",
            text_backbone="remoteclip",
            pair_feature_mode=args.pair_feature_mode,
            dinov2_model_path=str(args.dinov2_model_path),
            remoteclip_arch=args.remoteclip_arch,
            remoteclip_checkpoint=str(args.remoteclip_checkpoint),
            local_files_only=True,
            image_size=args.image_size,
        )
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    before = batch["before"].to(device)
    after = batch["after"].to(device)
    captions = [caption if isinstance(caption, str) else "" for caption in batch["caption"]]
    outputs = model(before, after, captions)
    positive_mask = torch.tensor(
        [[left == right for right in batch["pair_id"]] for left in batch["pair_id"]],
        dtype=torch.bool,
        device=device,
    )
    loss = symmetric_infonce_loss(outputs["change_embedding"], outputs["text_embedding"], positive_mask=positive_mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    finite_gradients = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
        if parameter.grad is not None
    )
    if not torch.isfinite(loss).item() or not finite_gradients:
        raise SystemExit("Non-finite loss or gradients detected during GPU smoke.")
    allocated_bytes = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "manifest": str(args.manifest),
        "batch_size": int(before.shape[0]),
        "unique_pair_ids_in_batch": len(set(batch["pair_id"])),
        "caption_row_count_in_batch": len(captions),
        "loss": float(loss.item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_gradients": finite_gradients,
        "change_embedding_shape": list(outputs["change_embedding"].shape),
        "text_embedding_shape": list(outputs["text_embedding"].shape),
        "patch_tokens_shape": list(outputs["patch_tokens"].shape),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_allocated_bytes": allocated_bytes,
        "gpu_peak_memory_bytes": peak_bytes,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "model_fingerprints": {
            "dinov2_model_path": path_fingerprint(args.dinov2_model_path),
            "remoteclip_checkpoint": path_fingerprint(args.remoteclip_checkpoint),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
