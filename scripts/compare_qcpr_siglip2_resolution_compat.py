#!/usr/bin/env python3
"""Compare fixed-resolution and NaFlex SigLIP-2 runtime on one exact gallery.

This is a compatibility evaluation only.  It restores the same temporal and
evidence adapter checkpoint on two independently loaded frozen backbones and
uses identical physical pairs and exact queries.  It does not train either
backbone or claim a scientific retrieval improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.manifest import group_rows_by_pair, load_exact_pair_rows
from qcpr_siglip2.data.runtime import encode_real_features
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return {"head": head, "worktree_clean": not bool(status.strip())}


def _load_config(path: Path) -> Siglip2TemporalConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config must contain a JSON object")
    fields = {
        name: payload[name]
        for name in Siglip2TemporalConfig.__dataclass_fields__
        if name in payload
    }
    return Siglip2TemporalConfig.from_dict(fields)


def _restore_adapter(
    model: Siglip2TemporalRetrievalModel, checkpoint: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain model_state")
    adapter_state = {
        key: value for key, value in state.items() if not key.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    non_backbone_missing = [key for key in missing if not key.startswith("backbone.")]
    if unexpected or non_backbone_missing:
        raise RuntimeError(
            "adapter checkpoint mismatch: "
            f"missing={non_backbone_missing[:8]} unexpected={unexpected[:8]}"
        )
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "restored_adapter_keys": len(adapter_state),
    }


def _metrics(scores: torch.Tensor, query_rows: list[dict[str, Any]], pair_ids: list[str]) -> dict[str, float]:
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    ranks: list[int] = []
    for row in query_rows:
        target = str(row["canonical_pair_id"])
        order = scores[len(ranks)].argsort(descending=True).tolist()
        ranks.append(order.index(pair_index[target]) + 1)
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "query_count": float(len(ranks)),
        "gallery_size": float(len(pair_ids)),
        "recall_at_1": float((rank_tensor <= 1).float().mean()),
        "recall_at_5": float((rank_tensor <= 5).float().mean()),
        "recall_at_10": float((rank_tensor <= 10).float().mean()),
        "mrr_full": float((1.0 / rank_tensor).mean()),
        "mean_rank": float(rank_tensor.mean()),
        "median_rank": float(rank_tensor.median()),
    }


def _select_rows(rows: list[dict[str, Any]], max_pairs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = group_rows_by_pair(rows)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    selected = ordered[:max_pairs]
    pair_rows = [group[0] for _, group in selected]
    query_rows = [group[0] for _, group in selected]
    if len(pair_rows) < 4:
        raise ValueError("compatibility comparison needs at least four pairs")
    return pair_rows, query_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naflex-model", required=True)
    parser.add_argument("--fixres-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--max-pairs", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = _git_state()
    if state["head"] != args.expected_code_sha or not state["worktree_clean"]:
        raise RuntimeError("runtime code state does not match expected clean SHA")
    release = Path(args.data_release)
    manifest = Path(args.development_manifest)
    checkpoint = Path(args.checkpoint)
    rows = load_exact_pair_rows(manifest, split="development")
    pair_rows, query_rows = _select_rows(rows, args.max_pairs)
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    config = _load_config(Path(args.config_path))
    result: dict[str, Any] = {
        "status": "COMPATIBILITY_SMOKE_ONLY",
        "evaluation_only": True,
        "code": state,
        "expected_code_sha": args.expected_code_sha,
        "release": str(release),
        "development_manifest": str(manifest),
        "development_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "pair_count": len(pair_rows),
        "query_count": len(query_rows),
        "pair_ids": pair_ids,
        "config": config.to_dict(),
        "models": {},
    }
    device = torch.device("cuda")
    outputs: dict[str, torch.Tensor] = {}
    for name, model_path, budget in (
        ("naflex_256", args.naflex_model, 256),
        ("fixres_256", args.fixres_model, None),
    ):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        backbone = Siglip2Backbone(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Siglip2TemporalRetrievalModel(backbone, config).to(device).eval()
        restore = _restore_adapter(model, checkpoint)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = encode_real_features(
                backbone,
                processor,
                pair_rows,
                query_rows,
                device,
                dtype=torch.bfloat16,
                no_grad=True,
                max_num_patches=budget,
                is_naflex=backbone.is_naflex,
            )
            forward = model.forward_from_features(
                features.frame_tokens,
                features.frame_embeddings,
                features.text_tokens,
                features.text_embeddings,
                features.text_mask,
                patch_valid_mask=features.patch_valid_mask,
                spatial_shapes=features.spatial_shapes,
                native_image_size=features.native_image_size,
                processed_patch_grid=features.processed_patch_grid,
                transform_hash=features.transform_hash,
            )
            scores = forward.score_matrix.float().cpu()
        torch.cuda.synchronize(device)
        outputs[name] = scores
        result["models"][name] = {
            "path": str(model_path),
            "runtime_class": backbone.runtime_class,
            "is_naflex": backbone.is_naflex,
            "native_patch_tokens": list(features.frame_tokens.shape),
            "score_shape": list(scores.shape),
            "metrics": _metrics(scores, query_rows, pair_ids),
            "adapter_restore": restore,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del model, backbone, processor, features, forward
        torch.cuda.empty_cache()
    naflex_scores = outputs["naflex_256"]
    fixres_scores = outputs["fixres_256"]
    result["score_matrix_max_abs_difference"] = float((naflex_scores - fixres_scores).abs().max())
    result["score_matrix_pearson"] = float(
        torch.corrcoef(torch.stack([naflex_scores.flatten(), fixres_scores.flatten()]))[0, 1]
    )
    result["top1_agreement"] = float(
        (naflex_scores.argmax(dim=1) == fixres_scores.argmax(dim=1)).float().mean()
    )
    (output / "resolution_compatibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolution_compatibility.sha256").write_text(
        f"{_sha256(output / 'resolution_compatibility.json')}  resolution_compatibility.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI failure contract
        print(f"RESOLUTION_COMPATIBILITY_FAILED: {exc}", file=sys.stderr)
        raise
