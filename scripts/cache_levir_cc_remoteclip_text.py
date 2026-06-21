from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.train_dino_pair_retrieval import choose_device, set_seed
from land_change_detection.models.dino_change_retriever import DINOChangeRetriever, DINOChangeRetrieverConfig
from land_change_detection.retrieval_cache import save_tensor_shard, write_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen RemoteCLIP caption features for LEVIR-CC.")
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remoteclip-arch", default="ViT-B-32")
    parser.add_argument("--remoteclip-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--captions-per-shard", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    rows = _read_jsonl(args.caption_manifest)
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="simple_patch",
            text_backbone="remoteclip",
            pair_feature_mode="change_fusion",
            remoteclip_arch=args.remoteclip_arch,
            remoteclip_checkpoint=str(args.remoteclip_checkpoint),
        )
    ).to(device)
    model.eval()
    text_encoder = model._load_text_encoder()

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    index_entries: dict[str, dict[str, str]] = {}
    shard_rows: list[dict] = []

    for start in range(0, len(rows), args.captions_per_shard):
        shard_batch = rows[start : start + args.captions_per_shard]
        tensors: dict[str, torch.Tensor] = {}
        shard_index: dict[str, dict[str, str]] = {}
        for batch_start in range(0, len(shard_batch), args.batch_size):
            mini = shard_batch[batch_start : batch_start + args.batch_size]
            captions = [str(row.get("caption") or "") for row in mini]
            with torch.no_grad():
                features = text_encoder(captions, device).detach().cpu().to(torch.float16)
            for offset, row in enumerate(mini):
                sample_id = str(row["sample_id"])
                key = f"{sample_id}__text"
                tensors[key] = features[offset]
                shard_index[sample_id] = {
                    "shard": "",
                    "key": key,
                    "pair_id": str(row["pair_id"]),
                    "feature_dim": int(features.shape[-1]),
                }
        shard_name = f"remoteclip_text_{start:05d}.safetensors"
        save_tensor_shard(shard_dir / shard_name, tensors, metadata={"dtype": "float16"})
        for sample_id, payload in shard_index.items():
            index_entries[sample_id] = {**payload, "shard": f"shards/{shard_name}"}
        shard_rows.append(
            {
                "path": f"shards/{shard_name}",
                "caption_count": len(shard_index),
                "sha256": hashlib.sha256(json.dumps(shard_index, sort_keys=True).encode("utf-8")).hexdigest(),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "cache_type": "levir_cc_remoteclip_text",
        "caption_manifest": str(args.caption_manifest),
        "captions": index_entries,
        "shards": shard_rows,
        "storage_dtype": "float16",
        "feature_dim": 512,
    }
    write_index(output_dir / "index.json", index_payload)
    print(json.dumps({"output_dir": str(output_dir), "caption_count": len(index_entries), "feature_dim": 512}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
