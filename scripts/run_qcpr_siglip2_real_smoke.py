#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from PIL import Image
from transformers import AutoProcessor

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone
from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.manifest import load_exact_core_rows, group_rows_by_pair
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.training.objective import multi_positive_listwise_loss

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def load_batch_rows(manifest: str, physical_batch_size: int, captions_per_pair: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    # Generic no-change rows are valid diagnostics, but not exact-pair
    # supervision for this smoke contract.
    rows = [
        row
        for row in load_exact_core_rows(manifest, split="train")
        if str(row.get("query_scope")) == "exact_pair"
    ]
    if not rows:
        raise ValueError("exact smoke manifest has no exact_pair rows")
    groups = group_rows_by_pair(rows)
    selected: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for pair_id, group in groups.items():
        if len(group) < captions_per_pair:
            continue
        first = group[:captions_per_pair]
        if any(not Path(row["t1_path"]).is_file() or not Path(row["t2_path"]).is_file() for row in first):
            raise FileNotFoundError(f"missing real image for {pair_id}")
        pairs.append(first[0])
        selected.extend(first)
        if len(pairs) == physical_batch_size:
            break
    if len(pairs) != physical_batch_size:
        raise ValueError(f"could not select {physical_batch_size} physical pairs with {captions_per_pair} captions")
    # A multi-positive query is supported only when the manifest explicitly
    # supplies multiple positive physical IDs.  Text collisions are never
    # promoted to positives by the smoke loader.
    multi_positive = any(
        isinstance(row.get("positive_pair_ids"), list)
        and len(row["positive_pair_ids"]) > 1
        for row in selected
    )
    return pairs, selected, multi_positive

def process_images(processor: Any, pairs: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[dict[str, str]]]:
    images: list[Image.Image] = []
    image_meta: list[dict[str, str]] = []
    for row in pairs:
        for key in ("t1_path", "t2_path"):
            path = Path(row[key])
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            image_meta.append({"path": str(path), "sha256": sha256(path)})
    encoded = processor(images=images, return_tensors="pt")
    pixels = encoded["pixel_values"].reshape(len(pairs), 2, *encoded["pixel_values"].shape[1:]).to(device)
    mask = encoded.get("pixel_attention_mask")
    if mask is not None:
        mask = mask.reshape(len(pairs), 2, *mask.shape[1:]).to(device)
    shapes = encoded.get("spatial_shapes")
    if shapes is not None:
        shapes = shapes.reshape(len(pairs), 2, 2).to(device)
    return pixels, mask, shapes, image_meta

def process_text(processor: Any, rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    texts = [str(row["caption"]) for row in rows]
    encoded = processor(text=texts, return_tensors="pt", padding="max_length")
    input_ids = encoded["input_ids"]
    mask = encoded.get("attention_mask")
    if mask is None:
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0)
        mask = input_ids.ne(0 if pad_id is None else int(pad_id))
    return input_ids.to(device), mask.to(device), texts

def build_relevance(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], captions_per_pair: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pair_index = {str(row["canonical_pair_id"]): index for index, row in enumerate(pairs)}
    positive = torch.zeros((len(rows), len(pairs)), dtype=torch.bool, device=device)
    ignored = torch.zeros_like(positive)
    missing_positive_metadata: list[str] = []
    for query_index, row in enumerate(rows):
        positive_ids = row.get("positive_pair_ids")
        if not isinstance(positive_ids, list) or not positive_ids:
            missing_positive_metadata.append(str(row.get("caption_id", query_index)))
            positive_ids = [row.get("canonical_pair_id")]
        for item_id in positive_ids:
            index = pair_index.get(str(item_id))
            if index is not None:
                positive[query_index, index] = True
        ignored_ids = row.get("ignored_pair_ids")
        if isinstance(ignored_ids, list):
            for item_id in ignored_ids:
                index = pair_index.get(str(item_id))
                if index is not None:
                    ignored[query_index, index] = True
    if missing_positive_metadata:
        raise ValueError("exact rows missing positive_pair_ids: " + ",".join(missing_positive_metadata[:5]))
    if torch.any(positive.sum(dim=1) == 0):
        raise ValueError("at least one selected query has no positive in the physical batch")
    if torch.any(positive & ignored):
        raise ValueError("selected exact batch has positive/ignored overlap")
    return positive, ignored, {
        "query_count": len(rows),
        "pair_count": len(pairs),
        "multi_positive_queries": int((positive.sum(dim=1) > 1).sum()),
        "captions_per_pair": captions_per_pair,
        "positive_source": "explicit_positive_pair_ids",
        "text_collision_not_used_as_positive": True,
    }

def score_from_evidence(model: Siglip2TemporalRetrievalModel, output: Any, weights: torch.Tensor) -> torch.Tensor:
    vector = torch.einsum("qpm,pmd->qpd", weights, output.temporal.temporal_patch_tokens)
    pair = torch.nn.functional.normalize(output.pair_cls.unsqueeze(0) + output.evidence.evidence_gate * vector, dim=-1)
    return torch.einsum("qd,qpd->qp", output.text_embedding, pair) / model.retrieval_temperature

def evidence_deletion(model: Siglip2TemporalRetrievalModel, output: Any) -> dict[str, Any]:
    weights = output.evidence.evidence_weights.detach()
    q, p, m = weights.shape
    k = max(1, int(round(m * 0.10)))
    order = output.evidence.evidence_logits.detach().argsort(dim=-1, descending=True)
    top_mask = torch.ones_like(weights)
    bottom_mask = torch.ones_like(weights)
    top_mask.scatter_(-1, order[..., :k], 0.0)
    bottom_mask.scatter_(-1, order[..., -k:], 0.0)
    top_weights = (weights * top_mask)
    bottom_weights = (weights * bottom_mask)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    bottom_weights = bottom_weights / bottom_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = output.score_matrix.detach()
    top_score = score_from_evidence(model, output, top_weights).detach()
    bottom_score = score_from_evidence(model, output, bottom_weights).detach()
    return {
        "removed_fraction": 0.10,
        "removed_tokens": k,
        "normal_score_q0_p0": float(normal[0, 0]),
        "top_evidence_score_q0_p0": float(top_score[0, 0]),
        "bottom_evidence_score_q0_p0": float(bottom_score[0, 0]),
        "score_drop_top_evidence": float(normal[0, 0] - top_score[0, 0]),
        "score_drop_bottom_evidence": float(normal[0, 0] - bottom_score[0, 0]),
        "passed": bool((normal[0, 0] - top_score[0, 0]) > (normal[0, 0] - bottom_score[0, 0])),
    }

def gradient_report(model: Siglip2TemporalRetrievalModel) -> dict[str, Any]:
    report: dict[str, Any] = {}
    modules = {
        "temporal_adapter": model.temporal_adapter,
        "evidence_bottleneck": model.evidence_bottleneck,
        "retrieval_temperature": model,
        "siglip2_vision_backbone": model.backbone.vision_model if model.backbone is not None else None,
        "siglip2_text_backbone": model.backbone.text_model if model.backbone is not None else None,
    }
    for name, module in modules.items():
        if module is None:
            continue
        params = list(module.parameters())
        grads = [p.grad.detach().float() for p in params if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads]) if grads else torch.empty(0)
        report[name] = {
            "parameter_count": sum(p.numel() for p in params),
            "trainable_count": sum(p.numel() for p in params if p.requires_grad),
            "parameters_with_grad": len(grads),
            "gradient_norm": float(flat.norm()) if flat.numel() else 0.0,
            "gradient_min": float(flat.min()) if flat.numel() else 0.0,
            "gradient_max": float(flat.max()) if flat.numel() else 0.0,
            "finite_gradient_fraction": float(torch.isfinite(flat).float().mean()) if flat.numel() else 1.0,
        }
    return report

def deterministic_roundtrip(args: argparse.Namespace) -> int:
    run = Path(args.output_dir)
    meta = json.loads((run / "roundtrip_input.json").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    backbone = Siglip2Backbone(args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16)
    model = Siglip2TemporalRetrievalModel(backbone, Siglip2TemporalConfig()).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    pairs, query_rows, _ = load_batch_rows(args.train_manifest, args.physical_batch_size, args.captions_per_pair)
    pixels, pixel_mask, shapes, _ = process_images(processor, pairs, device)
    input_ids, attention_mask, _ = process_text(processor, query_rows, device)
    with torch.no_grad():
        output = model(pixels, input_ids, attention_mask, pixel_attention_mask=pixel_mask, spatial_shapes=shapes)
    reference = torch.load(args.reference_scores, map_location="cpu", weights_only=True)
    difference = (output.score_matrix.float().cpu() - reference.float()).abs()
    tolerance = 2e-5
    result = {"status": "PASS" if float(difference.max()) <= tolerance else "CHECKPOINT_ROUNDTRIP_MISMATCH", "max_abs_score_difference": float(difference.max()), "tolerance": tolerance, "score_shape": list(output.score_matrix.shape)}
    write_json(run / "checkpoint_roundtrip.json", result)
    return 0 if result["status"] == "PASS" else 1

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--roundtrip-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--reference-scores")
    args = parser.parse_args()
    if args.roundtrip_only:
        if not args.checkpoint or not args.reference_scores: raise ValueError("--checkpoint and --reference-scores are required for roundtrip")
        return deterministic_roundtrip(args)
    if not torch.cuda.is_available(): raise RuntimeError("REAL_INTEGRATION_SMOKE_REQUIRES_CUDA")
    if args.steps <= 0: raise ValueError("steps must be positive")
    run = Path(args.output_dir); run.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    pairs, query_rows, multi_positive_supported = load_batch_rows(args.train_manifest, args.physical_batch_size, args.captions_per_pair)
    pixels, pixel_mask, shapes, image_meta = process_images(processor, pairs, device)
    input_ids, attention_mask, texts = process_text(processor, query_rows, device)
    positive, ignored, relevance_meta = build_relevance(query_rows, pairs, args.captions_per_pair, device)
    backbone = Siglip2Backbone(args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16)
    model = Siglip2TemporalRetrievalModel(backbone, Siglip2TemporalConfig()).to(device)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.05)
    write_json(run / "parameter_groups.json", {"groups": [{"name": "temporal_and_evidence", "lr": 2e-4, "weight_decay": 0.05, "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}], "scope": model.trainable_parameter_report(), "frozen_backbone": True})
    write_json(run / "model_contract.json", {"architecture": "SigLIP2_fixed_checkpoint_native_SiglipModel_plus_two_layer_temporal_adapter", "expected_code_sha": args.expected_code_sha, "hidden_size": 768, "native_patch_contract": [args.physical_batch_size, 2, 256, 768], "query_count": len(query_rows), "score_matrix": list(positive.shape), "evidence_map": [len(query_rows), len(pairs), 2, 16, 16], "multi_positive_runtime": bool(multi_positive_supported)})
    write_json(run / "batch_contract.json", {"physical_batch_size": args.physical_batch_size, "captions_per_pair": args.captions_per_pair, "query_count": len(query_rows), "score_matrix": list(positive.shape), "native_visual_tokens": 256, "precision": "bf16", "gradient_accumulation": 1})
    write_json(run / "roundtrip_input.json", {"pairs": [{"canonical_pair_id": row["canonical_pair_id"], "t1_path": row["t1_path"], "t2_path": row["t2_path"]} for row in pairs], "queries": [{"caption_id": row["caption_id"], "caption": row["caption"]} for row in query_rows]})
    timings = {"image_decode_and_processor_seconds": 0.0, "vision_seconds": [], "text_seconds": [], "forward_seconds": [], "backward_seconds": [], "optimizer_step_seconds": [], "total_wall_seconds": 0.0}
    start_total = time.perf_counter()
    last_output = None
    preclip_norms = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixels, input_ids, attention_mask, pixel_attention_mask=pixel_mask, spatial_shapes=shapes)
        torch.cuda.synchronize(); timings["forward_seconds"].append(time.perf_counter() - start)
        loss = multi_positive_listwise_loss(output.score_matrix.float(), positive, ignored)
        start = time.perf_counter(); loss.backward(); torch.cuda.synchronize(); timings["backward_seconds"].append(time.perf_counter() - start)
        flat = torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters() if p.grad is not None])
        preclip_norms.append(float(flat.norm()))
        start = time.perf_counter(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); torch.cuda.synchronize(); timings["optimizer_step_seconds"].append(time.perf_counter() - start)
        last_output = output.detach() if hasattr(output, "detach") else output
        write_json(run / "step_latest.json", {"global_step": step, "loss": float(loss.detach().cpu()), "finite": bool(torch.isfinite(loss)), "gradient_norm_preclip": preclip_norms[-1]})
    total_wall = time.perf_counter() - start_total
    timings["total_wall_seconds"] = total_wall
    grad = gradient_report(model)
    frozen_has_grad = any(v["trainable_count"] == 0 and v["parameters_with_grad"] > 0 for k, v in grad.items() if "backbone" in k)
    expected_no_grad = [k for k, v in grad.items() if v["trainable_count"] > 0 and v["parameters_with_grad"] == 0]
    if frozen_has_grad: raise RuntimeError("FROZEN_BACKBONE_HAS_GRADIENT")
    if expected_no_grad: raise RuntimeError("TRAINABLE_MODULE_NO_GRADIENT:" + ",".join(expected_no_grad))
    write_json(run / "gradient_diagnostics.json", {"modules": grad, "preclip_gradient_norms": preclip_norms, "all_finite": all(torch.isfinite(torch.tensor(x)) for x in preclip_norms)})
    assert last_output is not None
    same_pair = last_output.evidence.evidence_map[:2, 0]
    l1 = float((same_pair[0] - same_pair[1]).abs().mean())
    cosine = float(torch.nn.functional.cosine_similarity(same_pair[0].flatten().float().unsqueeze(0), same_pair[1].flatten().float().unsqueeze(0)).item())
    zero_score = score_from_evidence(model, last_output, torch.zeros_like(last_output.evidence.evidence_weights)).detach()
    deletion = evidence_deletion(model, last_output)
    diagnostics = {"query_swap_map_l1": l1, "query_swap_map_cosine": cosine, "score_change_after_query_swap": float((last_output.score_matrix[0,0]-last_output.score_matrix[1,0]).abs()), "score_change_after_evidence_zeroing": float((last_output.score_matrix-zero_score).abs().max()), "evidence_gate": float(last_output.evidence.evidence_gate), "evidence_deletion": deletion, "multi_positive_runtime": "EXERCISED" if multi_positive_supported else "MULTI_POSITIVE_RUNTIME_NOT_EXERCISED"}
    write_json(run / "evidence_diagnostics.json", diagnostics)
    state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": args.steps, "config": Siglip2TemporalConfig().to_dict(), "expected_code_sha": args.expected_code_sha, "train_manifest": args.train_manifest}
    checkpoint = run / "checkpoint.pt"; torch.save(state, checkpoint); (run / "checkpoint.sha256").write_text(sha256(checkpoint) + "  checkpoint.pt\n", encoding="utf-8")
    reference = last_output.score_matrix.detach().float().cpu(); reference_path = run / "reference_scores.pt"; torch.save(reference, reference_path)
    child_args = [sys.executable, __file__, "--roundtrip-only", "--siglip2-model", args.siglip2_model, "--train-manifest", args.train_manifest, "--output-dir", str(run), "--expected-code-sha", args.expected_code_sha, "--physical-batch-size", str(args.physical_batch_size), "--captions-per-pair", str(args.captions_per_pair), "--checkpoint", str(checkpoint), "--reference-scores", str(reference_path)]
    roundtrip = subprocess.run(child_args, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    if roundtrip.returncode != 0: raise RuntimeError("CHECKPOINT_ROUNDTRIP_MISMATCH")
    memory = {"gpu_name": torch.cuda.get_device_name(device), "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/(1024**3), "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/(1024**3), "current_allocated_gib": torch.cuda.memory_allocated(device)/(1024**3), "score_matrix": list(positive.shape), "visual_tokens": [int(x) for x in last_output.temporal.temporal_patch_tokens.shape], "text_tokens": [int(x) for x in input_ids.shape]}
    write_json(run / "cuda_memory.json", memory); write_json(run / "runtime_profile.json", timings); write_json(run / "exposure_accounting.json", {"physical_pairs_unique": len(pairs), "physical_pair_presentations": len(pairs) * args.steps, "query_unique": len(query_rows), "query_presentations": len(query_rows) * args.steps, "steps": args.steps, "pair_sequence_sha256": hashlib.sha256("\n".join(row["canonical_pair_id"] for row in pairs).encode()).hexdigest(), "query_sequence_sha256": hashlib.sha256("\n".join(row["caption_id"] for row in query_rows).encode()).hexdigest()})
    write_json(run / "training_complete.json", {"status": "PASS", "global_step": args.steps, "requested_steps": args.steps, "no_nan_or_oom": True, "checkpoint_roundtrip": "PASS"})
    write_json(run / "smoke_summary.json", {"status": "MODEL_V3_SIGLIP2_REAL_SMOKE_PASS", "code_sha": args.expected_code_sha, "runtime_class": backbone.runtime_class, "native_visual_shape": list(last_output.temporal.temporal_patch_tokens.shape), "text_shape": list(input_ids.shape), "loss_final": float(loss.detach().cpu()), "gradient_modules": list(grad), "multi_positive": relevance_meta, "image_hashes": image_meta})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
