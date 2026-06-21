from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.run_metadata import file_sha256, path_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local DINOv2 and official RemoteCLIP retrieval assets.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dinov2-model-path", type=Path, required=True)
    parser.add_argument("--remoteclip-checkpoint", type=Path, required=True)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dinov2_model_path.exists():
        raise SystemExit(f"Missing DINOv2 model path: {args.dinov2_model_path}")
    if not args.remoteclip_checkpoint.exists():
        raise SystemExit(f"Missing RemoteCLIP checkpoint: {args.remoteclip_checkpoint}")

    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="dinov2",
            text_backbone="remoteclip",
            pair_feature_mode="change_fusion",
            dinov2_model_path=str(args.dinov2_model_path),
            remoteclip_arch=args.remoteclip_arch,
            remoteclip_checkpoint=str(args.remoteclip_checkpoint),
            local_files_only=True,
            image_size=args.image_size,
        )
    )
    dino_hidden_size = int(model.visual_encoder.hidden_size)  # type: ignore[attr-defined]
    if dino_hidden_size != 768:
        raise SystemExit(f"Expected dinov2-base hidden size 768, got {dino_hidden_size}")

    images = torch.rand(2, 3, args.image_size, args.image_size)
    texts = ["new building appears", "water recedes near the field"]
    with torch.no_grad():
        outputs = model(images, images.flip(-1), texts)
        raw_text_encoder = model._load_text_encoder()
        raw_text = raw_text_encoder(texts, torch.device("cpu"))
    finite = {
        "change_embedding": bool(torch.isfinite(outputs["change_embedding"]).all().item()),
        "text_embedding": bool(torch.isfinite(outputs["text_embedding"]).all().item()),
        "raw_text_embedding": bool(torch.isfinite(raw_text).all().item()),
    }
    if not all(finite.values()):
        raise SystemExit(f"Non-finite embeddings detected: {finite}")

    normalized = {
        "change_embedding_norms": [float(value) for value in outputs["change_embedding"].norm(dim=-1)],
        "raw_text_embedding_norms": [float(value) for value in raw_text.norm(dim=-1)],
    }
    if any(abs(value - 1.0) > 1e-4 for value in normalized["change_embedding_norms"] + normalized["raw_text_embedding_norms"]):
        raise SystemExit(f"Unexpected embedding norms: {normalized}")

    projection_dim = int(model.text_projection.out_features) if model.text_projection is not None else None
    report = {
        "project_root": str(args.project_root) if args.project_root else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "remoteclip_arch": args.remoteclip_arch,
        "dinov2_model_path": path_fingerprint(args.dinov2_model_path),
        "remoteclip_checkpoint": {
            **(path_fingerprint(args.remoteclip_checkpoint) or {}),
            "sha256": file_sha256(args.remoteclip_checkpoint),
        },
        "dinov2_hidden_size": dino_hidden_size,
        "text_projection_out_features": projection_dim,
        "change_embedding_dim": int(outputs["change_embedding"].shape[-1]),
        "text_embedding_dim": int(outputs["text_embedding"].shape[-1]),
        "projection_dimensions_agree": projection_dim == int(outputs["change_embedding"].shape[-1]) == int(outputs["text_embedding"].shape[-1]),
        "finite_embeddings": finite,
        "normalized_embedding_norms": normalized,
    }
    if not report["projection_dimensions_agree"]:
        raise SystemExit(json.dumps(report, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
