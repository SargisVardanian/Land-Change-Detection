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
from qcpr_siglip2.data.manifest import load_exact_core_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import encode_real_images, encode_real_text
from qcpr_siglip2.evaluation.common_gallery import (
    canonical_pair_rows,
    exact_relevance_masks,
    metrics_by_query_group,
)
from qcpr_siglip2.evaluation.retrieval import full_gallery_metrics
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


def _encode_single_vector_gallery(
    model: Siglip2TemporalRetrievalModel,
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    image_batch_size: int,
    text_batch_size: int,
    max_num_patches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pair_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    visual_shapes: list[list[int]] = []
    for start in range(0, len(pair_rows), image_batch_size):
        image = encode_real_images(
            backbone,
            processor,
            pair_rows[start : start + image_batch_size],
            device,
            max_num_patches=max_num_patches,
            is_naflex=backbone.is_naflex,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            temporal = model.temporal_adapter(
                image.patch_tokens,
                image.pooled_embedding,
                patch_valid_mask=image.patch_valid_mask,
                spatial_shapes=image.spatial_shapes,
                native_image_size=image.native_image_size,
                processed_patch_grid=image.processed_patch_grid,
                transform_hash=image.transform_hash,
                token_coordinates=image.token_coordinates,
            )
        pair_embeddings.append(temporal.pair_cls.float().cpu())
        visual_shapes.append(list(image.patch_tokens.shape))
    for start in range(0, len(query_rows), text_batch_size):
        text = encode_real_text(
            backbone,
            processor,
            query_rows[start : start + text_batch_size],
            device,
        )
        text_embeddings.append(
            torch.nn.functional.normalize(text.pooled_embedding.float(), dim=-1).cpu()
        )
    return (
        torch.cat(pair_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        {"visual_batch_shapes": visual_shapes},
    )


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
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
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
    core_rows = load_exact_core_rows(manifest, split="development")
    query_rows = list(core_rows)
    pair_rows = canonical_pair_rows(core_rows)
    if args.max_pairs > 0:
        selected_ids = {
            str(row["canonical_pair_id"]) for row in pair_rows[: args.max_pairs]
        }
        pair_rows = pair_rows[: args.max_pairs]
        query_rows = [
            row for row in query_rows if str(row["canonical_pair_id"]) in selected_ids
        ]
    if len(pair_rows) < 4 or not query_rows:
        raise ValueError("compatibility comparison needs at least four pairs and queries")
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    positive, ignored = exact_relevance_masks(query_rows, pair_rows)
    exact_selector = torch.tensor(
        [row.get("query_scope") == "exact_pair" for row in query_rows],
        dtype=torch.bool,
    )
    if not exact_selector.any():
        raise ValueError("compatibility comparison has no primary exact queries")
    config = _load_config(Path(args.config_path))
    result: dict[str, Any] = {
        "status": "FULL_GALLERY_REGRESSION_PENDING",
        "evaluation_only": True,
        "code": state,
        "expected_code_sha": args.expected_code_sha,
        "release": str(release),
        "development_manifest": str(manifest),
        "development_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "pair_count": len(pair_rows),
        "query_count": len(query_rows),
        "exact_primary_query_count": int(exact_selector.sum()),
        "non_exact_diagnostic_query_count": int((~exact_selector).sum()),
        "pair_ids": pair_ids,
        "ordered_pair_ids_sha256": ordered_id_sha256(
            pair_rows, "canonical_pair_id"
        ),
        "ordered_query_ids_sha256": ordered_id_sha256(query_rows, "caption_id"),
        "config": config.to_dict(),
        "acceptance_contract": {
            "mrr_ratio_minimum": 0.85,
            "candidate_hit_at_10_max_absolute_regression": 0.02,
            "median_rank_ratio_maximum": 1.15,
            "defined_before_full_gallery_result": True,
        },
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
        pair_embeddings, text_embeddings, encoding_contract = (
            _encode_single_vector_gallery(
                model,
                backbone,
                processor,
                pair_rows,
                query_rows,
                device=device,
                image_batch_size=args.image_batch_size,
                text_batch_size=args.text_batch_size,
                max_num_patches=budget,
            )
        )
        scores = (
            text_embeddings @ pair_embeddings.transpose(0, 1)
            / float(model.retrieval_temperature.detach().cpu())
        ).masked_fill(ignored, -1.0e4)
        torch.cuda.synchronize(device)
        outputs[name] = scores
        result["models"][name] = {
            "path": str(model_path),
            "runtime_class": backbone.runtime_class,
            "is_naflex": backbone.is_naflex,
            "encoding_contract": encoding_contract,
            "pair_embedding_shape": list(pair_embeddings.shape),
            "text_embedding_shape": list(text_embeddings.shape),
            "score_shape": list(scores.shape),
            "metrics": full_gallery_metrics(
                scores[exact_selector], positive[exact_selector]
            ),
            "all_query_diagnostic_metrics": full_gallery_metrics(scores, positive),
            "per_source_metrics": metrics_by_query_group(
                scores, positive, query_rows
            ),
            "adapter_restore": restore,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del model, backbone, processor, pair_embeddings, text_embeddings
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
    naflex_metrics = result["models"]["naflex_256"]["metrics"]
    fixres_metrics = result["models"]["fixres_256"]["metrics"]
    mrr_ratio = float(naflex_metrics["mrr_full"] / fixres_metrics["mrr_full"])
    hit10_delta = float(
        naflex_metrics["candidate_hit_at_10"]
        - fixres_metrics["candidate_hit_at_10"]
    )
    median_ratio = float(
        naflex_metrics["median_rank"] / fixres_metrics["median_rank"]
    )
    accepted = bool(
        mrr_ratio >= 0.85 and hit10_delta >= -0.02 and median_ratio <= 1.15
    )
    result["acceptance_result"] = {
        "mrr_ratio": mrr_ratio,
        "candidate_hit_at_10_delta": hit10_delta,
        "median_rank_ratio": median_ratio,
        "passed": accepted,
    }
    result["status"] = (
        "NAFLEX_256_MATCHED_REGRESSION_PASS"
        if accepted
        else "NAFLEX_256_MATCHED_REGRESSION_FAIL"
    )
    torch.save(
        {
            "naflex_256_scores": naflex_scores,
            "fixres_256_scores": fixres_scores,
            "positive_mask": positive,
            "ignored_mask": ignored,
            "pair_ids": pair_ids,
            "query_ids": [str(row["caption_id"]) for row in query_rows],
        },
        output / "full_rankings.pt",
    )
    result["full_rankings_sha256"] = _sha256(output / "full_rankings.pt")
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
