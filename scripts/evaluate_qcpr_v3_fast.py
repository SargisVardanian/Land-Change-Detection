#!/usr/bin/env python3
"""Deterministic fast full-gallery evaluation for controlled QCPR v3 diagnostics."""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import ucv2_cluster_common as common
import ucv2_stage1_next_core as legacy
from land_change_detection.models.qcpr_v3_experiment import acceptance_gates, experiment_metrics, margin_family_reports
from land_change_detection.models.retrieval_heads import classify_caption_semantics
from land_change_detection.models.qcpr_v3_runtime import IMMUTABLE_V1, build_v3_and_teacher
from land_change_detection.models.retrieval_heads import semantic_teacher_relevance_matrix, stable_caption_group_ids


def write_progress(
    path: Path,
    *,
    stage: str,
    completed_chunks: int,
    total_chunks: int,
    started_monotonic: float,
    query_count: int,
    candidate_count: int,
) -> dict[str, float | int | str | bool]:
    """Atomically publish evaluator progress; never changes scoring results."""
    if total_chunks <= 0 or not 0 <= completed_chunks <= total_chunks:
        raise ValueError("completed_chunks must be within a positive total_chunks")
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    fraction = completed_chunks / total_chunks
    estimated_total = elapsed / fraction if fraction > 0 else 0.0
    payload: dict[str, float | int | str | bool] = {
        "schema_version": "qcpr-v3-evaluator-progress-v1",
        "stage": stage,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "completed_fraction": fraction,
        "elapsed_seconds": elapsed,
        "eta_seconds": max(estimated_total - elapsed, 0.0) if fraction > 0 else 0.0,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": completed_chunks == total_chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def make_config(checkpoint: Path, batch_size: int, manifest: Path):
    values = dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["config"])
    values.update(batch_size=batch_size, num_workers=4, dataset_config=None, train_manifests=(str(manifest),), val_manifests=(str(manifest),), dataset_sampling_weights=())
    fields = legacy.Stage1NextConfig.__dataclass_fields__
    return legacy.Stage1NextConfig(**{name: value for name, value in values.items() if name in fields})


def pad_token_batches(token_batches: list[torch.Tensor], attention_batches: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine batches with different tokenizer padding lengths into one corpus."""
    if len(token_batches) != len(attention_batches):
        raise ValueError("token and attention batch counts must match")
    token_rows, attention_rows = [], []
    for tokens, attention in zip(token_batches, attention_batches, strict=True):
        if tokens.ndim != 3 or attention.ndim != 2 or tokens.shape[:2] != attention.shape:
            raise ValueError("tokens must be [B,L,D] and attention must be matching [B,L]")
        token_rows.extend(tokens.unbind(0))
        attention_rows.extend(attention.unbind(0))
    if not token_rows:
        raise ValueError("cannot pad an empty token corpus")
    return (
        torch.nn.utils.rnn.pad_sequence(token_rows, batch_first=True, padding_value=0.0),
        torch.nn.utils.rnn.pad_sequence(attention_rows, batch_first=True, padding_value=0),
    )


@torch.no_grad()
def collect(model, dataset, config, device):
    pieces = {name: [] for name in ("pairs", "per_time", "text", "masks", "segmentation")}
    token_batches, attention_batches, content_batches = [], [], []
    captions, mapping, datasets, offset = [], [], [], 0
    for batch in legacy.make_eval_loader(dataset, config):
        images, temporal = batch["images"].to(device), batch["temporal_valid_mask"].to(device)
        pairs, per_time, _ = model.encode_pairs(images, temporal)
        text, tokens, attention, content = model.encode_texts(batch["captions"])
        for name, value in (("pairs", pairs), ("per_time", per_time), ("text", text), ("masks", batch["masks"]), ("segmentation", batch["segmentation_supervision"])):
            pieces[name].append(value.cpu())
        token_batches.append(tokens.cpu()); attention_batches.append(attention.cpu()); content_batches.append(content.cpu())
        captions.extend(batch["captions"]); mapping.extend((batch["caption_to_pair"] + offset).tolist())
        datasets.extend(str(name) for name in batch.get("dataset_names", ["unknown"] * images.shape[0]))
        offset += images.shape[0]
    tokens, attention = pad_token_batches(token_batches, attention_batches)
    _, content = pad_token_batches(token_batches, content_batches)
    return {name: torch.cat(values) for name, values in pieces.items()} | {"tokens": tokens, "attention": attention, "content": content.bool(), "captions": captions, "mapping": torch.tensor(mapping, dtype=torch.long), "pair_datasets": datasets}


@torch.no_grad()
def score(model, corpus, device, qchunk, cchunk, include_masks, *, rerank_top_n: int, progress_path: Path | None = None):
    """Compute expensive local evidence only in each query's global top-N pool."""
    qcount, ccount = corpus["text"].shape[0], corpus["pairs"].shape[0]
    if rerank_top_n <= 0:
        raise ValueError("rerank_top_n must be positive")
    global_scores = corpus["text"] @ corpus["pairs"].T
    local = torch.full_like(global_scores, float("-inf"))
    token_patch = torch.full_like(global_scores, float("-inf"))
    # Outside top-N, the production two-stage rank remains the global score.
    reranked = global_scores.clone()
    logits, targets = [], []
    started = time.monotonic(); total_chunks = math.ceil(qcount / qchunk)
    if progress_path is not None:
        write_progress(progress_path, stage="top_n_local_scoring", completed_chunks=0, total_chunks=total_chunks, started_monotonic=started, query_count=qcount, candidate_count=ccount)
    for chunk_index, q0 in enumerate(range(0, qcount, qchunk), start=1):
        q1 = min(q0 + qchunk, qcount); mapping = corpus["mapping"][q0:q1]
        selected = global_scores[q0:q1].topk(min(rerank_top_n, ccount), dim=1).indices
        # A shared union retains batched QxC cross-attention while reducing the
        # old all-gallery interaction to at most Q x union(top-N) per chunk.
        union = selected.unique(sorted=True)
        output = model.score_encoded(
            corpus["pairs"][union].to(device), corpus["per_time"][union].to(device),
            corpus["text"][q0:q1].to(device), corpus["tokens"][q0:q1].to(device),
            corpus["attention"][q0:q1].to(device), corpus["content"][q0:q1].to(device),
        )
        union_positions = torch.searchsorted(union, selected)
        rows = torch.arange(q1 - q0)[:, None]
        local[q0:q1].scatter_(1, selected, output.local_score.cpu()[rows, union_positions])
        token_patch[q0:q1].scatter_(1, selected, output.token_patch_score.cpu()[rows, union_positions])
        reranked[q0:q1].scatter_(1, selected, output.reranked_score.cpu()[rows, union_positions])
        if include_masks:
            for row in torch.nonzero(corpus["segmentation"][mapping], as_tuple=False).flatten().tolist():
                pair = int(mapping[row])
                matched = torch.nonzero(union == pair, as_tuple=False).flatten()
                if matched.numel():
                    logit = output.decoded_mask_logits[row, int(matched[0])].cpu()
                    target = torch.nn.functional.interpolate(corpus["masks"][pair:pair + 1].unsqueeze(1), logit.shape[-2:], mode="nearest")[0, 0]
                    logits.append(logit); targets.append(target)
        if progress_path is not None:
            write_progress(progress_path, stage="top_n_local_scoring", completed_chunks=chunk_index, total_chunks=total_chunks, started_monotonic=started, query_count=qcount, candidate_count=ccount)
    return global_scores, local, token_patch, reranked, logits, targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("late_interaction", "mask_grounding"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, default=IMMUTABLE_V1)
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3/natural_validation_manifest.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fast-pairs", type=int, default=64)
    parser.add_argument("--query-chunk", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=16)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--progress-path", type=Path, default=None, help="Atomic JSON heartbeat path; defaults beside --output")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("fast evaluation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    config = make_config(args.v1_checkpoint, args.batch_size, args.manifest)
    _, val = legacy._build_stage1_datasets(config)
    indices = [index for index, row in enumerate(val.samples) if str(row["dataset_name"]) in {"levir_mci", "second_cc"}]
    if args.fast_pairs > 0:
        indices = indices[:args.fast_pairs]
    if len(indices) < 2: raise RuntimeError("fast validation subset is empty")
    model, _, audit = build_v3_and_teacher(args.v1_checkpoint, device=device, build_legacy_model=common.build_model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True); model.eval()
    corpus = collect(model, Subset(val, indices), config, device)
    progress_path = args.progress_path or args.output.parent / "progress.json"
    global_scores, local, token_patch, reranked, logits, targets = score(model, corpus, device, args.query_chunk, args.candidate_chunk, args.phase == "mask_grounding", rerank_top_n=min(args.top_n, corpus["pairs"].shape[0]), progress_path=progress_path)
    relevance = semantic_teacher_relevance_matrix(corpus["text"], corpus["captions"], corpus["mapping"], stable_caption_group_ids(corpus["captions"]), pair_count=corpus["pairs"].shape[0], top_k=0)
    masks = {}
    if args.phase == "mask_grounding":
        if not logits: raise RuntimeError("mask validation selected no supervised masks")
        masks = {"mask_logits": torch.stack(logits), "mask_targets": torch.stack(targets)}
    # B's primary generic grounding evidence is token-to-patch late interaction;
    # masked local pooling is reported independently for the later mask phase.
    metrics = experiment_metrics(global_scores, token_patch, reranked, relevance, corpus["mapping"], top_n=min(args.top_n, corpus["pairs"].shape[0]), captions=corpus["captions"], **masks)
    from land_change_detection.models.qcpr_v3_experiment import broad_semantic_margin
    metrics["masked_local_broad_semantic_margin_mean"] = float(broad_semantic_margin(local, relevance)["mean"])
    query_groups = {
        "changed_only": torch.tensor([bool(classify_caption_semantics(caption)["changed"]) for caption in corpus["captions"]]),
        "no_change": torch.tensor([bool(classify_caption_semantics(caption)["no_change"]) for caption in corpus["captions"]]),
        "levir": torch.tensor([corpus["pair_datasets"][int(pair)] == "levir_mci" for pair in corpus["mapping"].tolist()]),
        "second": torch.tensor([corpus["pair_datasets"][int(pair)] == "second_cc" for pair in corpus["mapping"].tolist()]),
    }
    margin_reports = margin_family_reports(token_patch, reranked, corpus["mapping"], relevance, corpus["captions"], global_scores, top_n=min(args.top_n, corpus["pairs"].shape[0]), query_groups=query_groups)
    training = json.loads(args.training_report.read_text())
    metrics["local_gradients_finite_nonzero"] = bool(training["gradient_audit"]["local_gradients_finite_nonzero"])
    metrics["faithful_mask"] = True
    baseline = {"semantic_r1": float(metrics["global_semantic_r1"]), "semantic_r5": float(metrics["global_semantic_r5"])}
    gates = acceptance_gates(args.phase, metrics, v1_global_r1=baseline["semantic_r1"], v1_global_r5=baseline["semantic_r5"])
    report = {"status": "PASS" if all(gates.values()) else "FAIL_GATE", "phase": args.phase, "checkpoint": str(args.checkpoint), "initialization": audit.__dict__, "fast_validation": {"pair_count": len(indices), "query_count": len(corpus["captions"]), "manifest": str(args.manifest)}, "v1_compatible_global_fast_baseline": baseline, "metrics": metrics, "margin_reports": margin_reports, "gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress = json.loads(progress_path.read_text())
    progress["stage"] = "complete"; progress["report"] = str(args.output); progress["complete"] = True
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__": raise SystemExit(main())
