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
    parser = argparse.ArgumentParser(description="Cache frozen DINOv2 LEVIR-CC pair tokens in shard files.")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dinov2-model-path", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pairs-per-shard", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(path: str, project_root: Path | None) -> str:
    raw = Path(path)
    if raw.is_absolute() or project_root is None:
        return str(raw)
    return str((project_root / raw).resolve())


def _load_image(path: str, image_size: int) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _shard_hash(entries: dict[str, dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    rows = _read_jsonl(args.pair_manifest)
    project_root = args.project_root.resolve() if args.project_root is not None else None
    device = choose_device(args.device)
    model = DINOChangeRetriever(
        DINOChangeRetrieverConfig(
            visual_backbone="dinov2",
            text_backbone="simple_text",
            pair_feature_mode="signed_delta",
            dinov2_model_path=str(args.dinov2_model_path),
            local_files_only=True,
            image_size=args.image_size,
        )
    ).to(device)
    model.eval()

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    index_entries: dict[str, dict[str, str]] = {}
    shard_rows: list[dict] = []

    for start in range(0, len(rows), args.pairs_per_shard):
        shard_rows_batch = rows[start : start + args.pairs_per_shard]
        tensors: dict[str, torch.Tensor] = {}
        shard_pairs: dict[str, dict[str, str]] = {}
        for batch_start in range(0, len(shard_rows_batch), args.batch_size):
            mini = shard_rows_batch[batch_start : batch_start + args.batch_size]
            before = torch.stack(
                [_load_image(_resolve(str(row["before_path"]), project_root), args.image_size) for row in mini]
            ).to(device)
            after = torch.stack(
                [_load_image(_resolve(str(row["after_path"]), project_root), args.image_size) for row in mini]
            ).to(device)
            with torch.no_grad():
                before_patch, before_cls = model.encode_image_tokens(before)
                after_patch, after_cls = model.encode_image_tokens(after)
            before_tokens = torch.cat([before_cls.unsqueeze(1), before_patch], dim=1).detach().cpu().to(torch.float16)
            after_tokens = torch.cat([after_cls.unsqueeze(1), after_patch], dim=1).detach().cpu().to(torch.float16)
            for offset, row in enumerate(mini):
                pair_id = str(row["pair_id"])
                before_key = f"{pair_id}__before"
                after_key = f"{pair_id}__after"
                tensors[before_key] = before_tokens[offset]
                tensors[after_key] = after_tokens[offset]
                shard_pairs[pair_id] = {
                    "before_key": before_key,
                    "after_key": after_key,
                    "shape": list(before_tokens[offset].shape),
                }

        shard_name = f"dinov2_pairs_{start:05d}.safetensors"
        save_tensor_shard(shard_dir / shard_name, tensors, metadata={"dtype": "float16"})
        for pair_id, payload in shard_pairs.items():
            index_entries[pair_id] = {"shard": f"shards/{shard_name}", **payload}
        shard_rows.append({"path": f"shards/{shard_name}", "pair_count": len(shard_pairs), "sha256": _shard_hash(shard_pairs)})

    sample_pair = rows[0]
    sample_id = str(sample_pair["pair_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "cache_type": "levir_cc_dinov2_pair_tokens",
        "pair_manifest": str(args.pair_manifest),
        "pairs": index_entries,
        "shards": shard_rows,
        "expected_token_shape": [257, 768],
        "storage_dtype": "float16",
        "validated_pair_id": sample_id,
    }
    write_index(output_dir / "index.json", index_payload)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(index_entries), "validated_pair_id": sample_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
