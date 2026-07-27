#!/usr/bin/env python3
"""Fixed-split QCPR v3.1 frozen encoder ablation.

The learned QCPR global pair encoder is the production anchor. RemoteCLIP,
GeoRSCLIP and SigLIP2 are evaluated with an explicitly labelled zero-shot
signed temporal-delta contract. Their dense-grid metadata is also recorded for
the later grounding-backbone decision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import qcpr_v3_data_compat as data_compat
from land_change_detection.models.qcpr_v3 import QCPRV3Config
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model
from land_change_detection.models.qcpr_v31_encoder_ablation import (
    load_global_retrieval_modules_strict,
    metrics_by_query_groups,
    pooled_feature_tensor,
    replacement_gate,
    score_zero_shot_temporal_delta,
    sha256_file,
    stable_fingerprint,
)
from land_change_detection.models.retrieval_heads import (
    classify_caption_semantics,
    semantic_teacher_relevance_matrix,
    stable_caption_group_ids,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.started = time.monotonic()
        self.stage: str | None = None
        self.stage_started = self.started

    def write(self, stage: str, completed: int, total: int, **extra: Any) -> None:
        now = time.monotonic()
        if stage != self.stage:
            self.stage = stage
            self.stage_started = now
        elapsed = max(now - self.started, 0.0)
        stage_elapsed = max(now - self.stage_started, 0.0)
        fraction = completed / total if total else 0.0
        eta = stage_elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        atomic_json(
            self.path,
            {
                "schema_version": "qcpr-v31-encoder-ablation-progress-v1",
                "stage": stage,
                "completed": completed,
                "total": total,
                "completed_fraction": fraction,
                "elapsed_seconds": elapsed,
                "stage_elapsed_seconds": stage_elapsed,
                "eta_seconds": eta,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "complete": bool(total and completed == total and stage == "complete"),
                **extra,
            },
        )


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str], torch.Tensor]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("retrieval manifest is empty")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError("retrieval manifest has duplicate pair IDs")
    captions: list[str] = []
    mapping: list[int] = []
    for pair_index, row in enumerate(rows):
        row_captions = [str(value) for value in row.get("captions", [])]
        if not row_captions:
            raise RuntimeError(f"pair {row['pair_id']} has no captions")
        captions.extend(row_captions)
        mapping.extend([pair_index] * len(row_captions))
    return rows, captions, torch.tensor(mapping, dtype=torch.long)


def make_current_config(payload: dict[str, Any], manifest: Path, batch_size: int):
    values = dict(payload["data_config"])
    values.update(
        batch_size=batch_size,
        num_workers=4,
        dataset_config=None,
        train_manifests=(str(manifest),),
        val_manifests=(str(manifest),),
        dataset_sampling_weights=(),
        max_captions_per_pair=0,
    )
    fields = data_compat.Stage1NextConfig.__dataclass_fields__
    return data_compat.Stage1NextConfig(
        **{name: value for name, value in values.items() if name in fields}
    )


@torch.no_grad()
def encode_current(
    checkpoint: Path,
    manifest: Path,
    batch_size: int,
    device: torch.device,
    progress: Progress,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = make_current_config(payload, manifest, batch_size)
    _, dataset = data_compat.build_datasets(config)
    expected_rows, expected_captions, expected_mapping = load_manifest(manifest)
    if len(dataset) != len(expected_rows):
        raise RuntimeError("current dataset pair count differs from manifest")
    actual_ids = [str(sample["pair_id"]) for sample in dataset.samples]
    expected_ids = [str(row["pair_id"]) for row in expected_rows]
    if actual_ids != expected_ids:
        raise RuntimeError("current dataset pair order differs from manifest")
    model = build_clean_v3_model(
        QCPRV3BackboneConfig(**payload["backbone_config"]),
        device=device,
        grounding_config=QCPRV3Config(**payload["grounding_config"]),
    )
    global_load = load_global_retrieval_modules_strict(model, payload["model"])
    model.eval()
    pairs: list[torch.Tensor] = []
    base_text: list[torch.Tensor] = []
    adapted_text: list[torch.Tensor] = []
    captions: list[str] = []
    mapping: list[int] = []
    offset = 0
    loader = data_compat.make_eval_loader(dataset, config)
    for index, batch in enumerate(loader, 1):
        pair, _, _ = model.encode_pairs(
            batch["images"].to(device), batch["temporal_valid_mask"].to(device)
        )
        base, adapted, _, _, _ = model.encode_texts(batch["captions"])
        pairs.append(pair.cpu())
        base_text.append(base.cpu())
        adapted_text.append(adapted.cpu())
        captions.extend(batch["captions"])
        mapping.extend((batch["caption_to_pair"] + offset).tolist())
        offset += int(batch["images"].shape[0])
        progress.write("current_jina_universat", index, len(loader))
    actual_mapping = torch.tensor(mapping, dtype=torch.long)
    if captions != expected_captions or not torch.equal(actual_mapping, expected_mapping):
        raise RuntimeError("current query order or caption-to-pair mapping differs from manifest")
    return (
        torch.cat(base_text),
        torch.cat(adapted_text),
        torch.cat(pairs),
        captions,
        actual_mapping,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "strict_global_path_load": True,
            "global_load_audit": global_load,
            "grounder_load": "EXCLUDED_ARCHITECTURE_VERSIONED_AND_UNUSED",
            "score_contract": "learned_qcpr_global_pair_embedding",
            "architecture_compatible_replacement": True,
            "dense_grid": [32, 32],
            "dense_dim": 768,
        },
    )


def image_batches(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        before = [Image.open(row["t1_path"]).convert("RGB") for row in chunk]
        after = [Image.open(row["t2_path"]).convert("RGB") for row in chunk]
        yield start // batch_size + 1, before, after


@torch.no_grad()
def encode_open_clip(
    label: str,
    model_name: str,
    checkpoint: Path,
    rows: list[dict[str, Any]],
    captions: list[str],
    batch_size: int,
    device: torch.device,
    progress: Progress,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    text_parts = []
    for start in range(0, len(captions), batch_size * 4):
        text_parts.append(model.encode_text(tokenizer(captions[start : start + batch_size * 4]).to(device)).cpu())
    before_parts, after_parts = [], []
    total = math.ceil(len(rows) / batch_size)
    for index, before, after in image_batches(rows, batch_size):
        before_parts.append(model.encode_image(torch.stack([preprocess(image) for image in before]).to(device)).cpu())
        after_parts.append(model.encode_image(torch.stack([preprocess(image) for image in after]).to(device)).cpu())
        progress.write(label, index, total)
    dense_grid = [7, 7] if model_name == "ViT-B-32" else None
    return (
        torch.cat(text_parts),
        torch.cat(before_parts),
        torch.cat(after_parts),
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "strict_load": True,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "model_name": model_name,
            "score_contract": "zero_shot_cosine_text_vs_normalized_after_minus_before",
            "architecture_compatible_replacement": False,
            "dense_grid": dense_grid,
            "tokenizer_context_length": int(getattr(model, "context_length", 77)),
        },
    )


@torch.no_grad()
def encode_siglip2(
    model_path: Path,
    rows: list[dict[str, Any]],
    captions: list[str],
    batch_size: int,
    device: torch.device,
    progress: Progress,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    text_parts = []
    for start in range(0, len(captions), batch_size * 4):
        inputs = processor(
            text=captions[start : start + batch_size * 4],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        output = model.get_text_features(**inputs)
        text_parts.append(
            pooled_feature_tensor(
                output,
                expected_batch=int(inputs["input_ids"].shape[0]),
            ).cpu()
        )
    before_parts, after_parts = [], []
    total = math.ceil(len(rows) / batch_size)
    for index, before, after in image_batches(rows, batch_size):
        before_inputs = processor(images=before, return_tensors="pt")
        after_inputs = processor(images=after, return_tensors="pt")
        before_output = model.get_image_features(
            **{k: v.to(device) for k, v in before_inputs.items()}
        )
        after_output = model.get_image_features(
            **{k: v.to(device) for k, v in after_inputs.items()}
        )
        before_parts.append(
            pooled_feature_tensor(before_output, expected_batch=len(before)).cpu()
        )
        after_parts.append(
            pooled_feature_tensor(after_output, expected_batch=len(after)).cpu()
        )
        progress.write("siglip2", index, total)
    config_path = model_path / "config.json"
    return (
        torch.cat(text_parts),
        torch.cat(before_parts),
        torch.cat(after_parts),
        {
            "model_path": str(model_path),
            "config_sha256": sha256_file(config_path),
            "local_files_only": True,
            "score_contract": "zero_shot_cosine_text_vs_normalized_after_minus_before",
            "architecture_compatible_replacement": False,
            "dense_grid": [16, 16],
            "dense_dim": int(model.config.vision_config.hidden_size),
            "image_size": int(model.config.vision_config.image_size),
            "patch_size": int(model.config.vision_config.patch_size),
            "text_max_length": int(model.config.text_config.max_position_embeddings),
        },
    )


def query_groups(captions: list[str], rows: list[dict[str, Any]], mapping: torch.Tensor):
    return {
        "changed_only": torch.tensor(
            [bool(classify_caption_semantics(caption)["changed"]) for caption in captions]
        ),
        "no_change": torch.tensor(
            [bool(classify_caption_semantics(caption)["no_change"]) for caption in captions]
        ),
        "levir": torch.tensor(
            [rows[int(pair)]["dataset_name"] == "levir_mci" for pair in mapping]
        ),
        "second": torch.tensor(
            [rows[int(pair)]["dataset_name"] == "second_cc" for pair in mapping]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--remoteclip-checkpoint", type=Path, required=True)
    parser.add_argument("--siglip2-model", type=Path, required=True)
    parser.add_argument("--georsclip-checkpoint", type=Path)
    parser.add_argument("--georsclip-model-name", default="ViT-L-14")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("encoder ablation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = Progress(args.output_dir / "progress.json")
    rows, captions, mapping = load_manifest(args.manifest)
    progress.write("initializing", 0, 1)
    base_text, adapted_text, current_pairs, actual_captions, actual_mapping, current_meta = encode_current(
        args.current_checkpoint, args.manifest, args.batch_size, device, progress
    )
    if actual_captions != captions or not torch.equal(actual_mapping, mapping):
        raise RuntimeError("current encoder order contract failed")
    relevance = semantic_teacher_relevance_matrix(
        base_text,
        captions,
        mapping,
        stable_caption_group_ids(captions),
        pair_count=len(rows),
        top_k=0,
    ).bool()
    groups = query_groups(captions, rows, mapping)
    current_scores = torch.nn.functional.normalize(adapted_text, dim=-1) @ current_pairs.T
    reports: dict[str, Any] = {
        "current_jina_universat": {
            "metadata": current_meta,
            "metrics": metrics_by_query_groups(current_scores, relevance, mapping, groups),
        }
    }
    score_artifacts: dict[str, torch.Tensor] = {"current_jina_universat": current_scores}
    encoders: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]]]] = [
        (
            "remoteclip_vit_b_32",
            lambda: encode_open_clip(
                "remoteclip_vit_b_32",
                "ViT-B-32",
                args.remoteclip_checkpoint,
                rows,
                captions,
                args.batch_size,
                device,
                progress,
            ),
        ),
        (
            "siglip2_base_patch16_256",
            lambda: encode_siglip2(
                args.siglip2_model, rows, captions, args.batch_size, device, progress
            ),
        ),
    ]
    if args.georsclip_checkpoint is not None:
        encoders.append(
            (
                "georsclip",
                lambda: encode_open_clip(
                    "georsclip",
                    args.georsclip_model_name,
                    args.georsclip_checkpoint,
                    rows,
                    captions,
                    args.batch_size,
                    device,
                    progress,
                ),
            )
        )
    anchor_all = reports["current_jina_universat"]["metrics"]["all"]
    for label, encode in encoders:
        text, before, after, metadata = encode()
        scores = score_zero_shot_temporal_delta(text, before, after)
        metrics = metrics_by_query_groups(scores, relevance, mapping, groups)
        reports[label] = {
            "metadata": metadata,
            "metrics": metrics,
            "production_replacement_gate": {
                "status": "NOT_COMPARABLE",
                "reason": (
                    "single-image zero-shot temporal delta has no trained bi-temporal pair "
                    "head; it cannot replace the learned production pair encoder from this "
                    "ablation alone"
                ),
                "hypothetical_metric_gate": replacement_gate(anchor_all, metrics["all"]),
            },
        }
        score_artifacts[label] = scores
    pair_ids = [str(row["pair_id"]) for row in rows]
    fingerprints = {
        "manifest_sha256": sha256_file(args.manifest),
        "pair_order_sha256": stable_fingerprint(pair_ids),
        "query_order_sha256": stable_fingerprint(captions),
        "caption_to_pair_sha256": stable_fingerprint(mapping.tolist()),
        "semantic_positive_mask_sha256": stable_fingerprint(relevance.to(torch.uint8).tolist()),
        "preprocessing_sha256": stable_fingerprint(
            [row.get("preprocessing_fingerprint") for row in rows]
        ),
    }
    report = {
        "schema_version": "qcpr-v31-encoder-ablation-v1",
        "status": "PASS",
        "scientific_contract": {
            "production_anchor": "current_jina_universat",
            "semantic_relevance": "text-derived pseudo relevance, not image ground truth",
            "exact_pair_metrics": "diagnostic only",
            "alternative_score": "zero-shot signed temporal delta, not architecture-compatible",
            "global_replacement_rule": "requires a separately trained compatible temporal adapter",
        },
        "environment": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
        },
        "manifest": str(args.manifest),
        "pair_count": len(rows),
        "query_count": len(captions),
        "fingerprints": fingerprints,
        "encoders": reports,
    }
    atomic_json(args.output_dir / "encoder_ablation_report.json", report)
    torch.save(
        {
            "scores": score_artifacts,
            "relevance": relevance,
            "mapping": mapping,
            "fingerprints": fingerprints,
        },
        args.output_dir / "encoder_ablation_scores.pt",
    )
    progress.write(
        "complete",
        1,
        1,
        report=str(args.output_dir / "encoder_ablation_report.json"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
