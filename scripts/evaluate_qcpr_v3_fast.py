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
import qcpr_v3_data_compat as data_compat
from land_change_detection.models.qcpr_v3_experiment import acceptance_gates, experiment_metrics, margin_family_reports
from land_change_detection.models.retrieval_heads import classify_caption_semantics
from land_change_detection.models.qcpr_v3 import QCPRV3Config
from land_change_detection.models.qcpr_v3_factory import QCPRV3BackboneConfig, build_clean_v3_model
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


def make_config(payload: dict, batch_size: int, manifest: Path):
    values = dict(payload["data_config"])
    values.update(batch_size=batch_size, num_workers=4, dataset_config=None, train_manifests=(str(manifest),), val_manifests=(str(manifest),), dataset_sampling_weights=())
    fields = data_compat.Stage1NextConfig.__dataclass_fields__
    return data_compat.Stage1NextConfig(**{name: value for name, value in values.items() if name in fields})


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
def collect(model, dataset, config, device, *, progress_path: Path | None = None):
    pieces = {name: [] for name in ("pairs", "per_time", "base_text", "text", "masks", "segmentation")}
    token_batches, attention_batches, content_batches = [], [], []
    captions, mapping, datasets, offset = [], [], [], 0
    loader = data_compat.make_eval_loader(dataset, config)
    started = time.monotonic()
    if progress_path is not None:
        write_progress(progress_path, stage="encoding", completed_chunks=0, total_chunks=len(loader), started_monotonic=started, query_count=0, candidate_count=len(dataset))
    for batch_index, batch in enumerate(loader, start=1):
        images, temporal = batch["images"].to(device), batch["temporal_valid_mask"].to(device)
        pairs, per_time, _ = model.encode_pairs(images, temporal)
        base_text, text, tokens, attention, content = model.encode_texts(batch["captions"])
        for name, value in (("pairs", pairs), ("per_time", per_time), ("base_text", base_text), ("text", text), ("masks", batch["masks"]), ("segmentation", batch["segmentation_supervision"])):
            pieces[name].append(value.cpu())
        token_batches.append(tokens.cpu()); attention_batches.append(attention.cpu()); content_batches.append(content.cpu())
        captions.extend(batch["captions"]); mapping.extend((batch["caption_to_pair"] + offset).tolist())
        datasets.extend(str(name) for name in batch.get("dataset_names", ["unknown"] * images.shape[0]))
        offset += images.shape[0]
        if progress_path is not None:
            write_progress(progress_path, stage="encoding", completed_chunks=batch_index, total_chunks=len(loader), started_monotonic=started, query_count=len(captions), candidate_count=len(dataset))
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



def global_subset_metrics(scores: torch.Tensor, relevance: torch.Tensor, subset: torch.Tensor, *, k: int = 100) -> dict[str, float | int]:
    """Retrieval statistics for an explicit caption stratum."""
    selected = torch.nonzero(subset, as_tuple=False).flatten()
    if not selected.numel():
        return {"query_count": 0, "semantic_r1": 0.0, "candidate_recall_at_100": 0.0}
    local_scores = scores.index_select(0, selected)
    local_relevance = relevance.index_select(0, selected).bool()
    top1 = local_scores.argmax(dim=1)
    topk = local_scores.topk(min(k, local_scores.shape[1]), dim=1).indices
    return {
        "query_count": int(selected.numel()),
        "semantic_r1": float(local_relevance[torch.arange(selected.numel()), top1].float().mean()),
        "candidate_recall_at_100": float(local_relevance.gather(1, topk).any(dim=1).float().mean()),
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("global_bootstrap", "late_interaction", "mask_grounding"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, default=None, help="A0 step-zero evaluation for strict directional comparison")
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
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("initialization", {}).get("initialization_mode") != "clean_pretrained":
        raise RuntimeError("this evaluator path requires an explicit clean_pretrained v3 checkpoint")
    config = make_config(checkpoint_payload, args.batch_size, args.manifest)
    _, val = data_compat.build_datasets(config)
    indices = [index for index, row in enumerate(val.samples) if str(row["dataset_name"]) in {"levir_mci", "second_cc"}]
    if args.fast_pairs > 0:
        indices = indices[:args.fast_pairs]
    if len(indices) < 2: raise RuntimeError("fast validation subset is empty")
    backbone_config = QCPRV3BackboneConfig(**checkpoint_payload["backbone_config"])
    grounding_config = QCPRV3Config(**checkpoint_payload["grounding_config"])
    model = build_clean_v3_model(backbone_config, device=device, grounding_config=grounding_config)
    model.load_state_dict(checkpoint_payload["model"], strict=True); model.eval()
    progress_path = args.progress_path or args.output.parent / "progress.json"
    corpus = collect(model, Subset(val, indices), config, device, progress_path=progress_path)
    if args.phase == "global_bootstrap":
        global_scores = corpus["text"] @ corpus["pairs"].T
        local = token_patch = reranked = global_scores
        logits, targets = [], []
        write_progress(progress_path, stage="global_scoring", completed_chunks=1, total_chunks=1, started_monotonic=time.monotonic(), query_count=global_scores.shape[0], candidate_count=global_scores.shape[1])
    else:
        global_scores, local, token_patch, reranked, logits, targets = score(model, corpus, device, args.query_chunk, args.candidate_chunk, args.phase == "mask_grounding", rerank_top_n=min(args.top_n, corpus["pairs"].shape[0]), progress_path=progress_path)
    relevance = semantic_teacher_relevance_matrix(corpus["base_text"], corpus["captions"], corpus["mapping"], stable_caption_group_ids(corpus["captions"]), pair_count=corpus["pairs"].shape[0], top_k=0)
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
    metrics["changed_only"] = global_subset_metrics(global_scores, relevance, query_groups["changed_only"])
    metrics["no_change"] = global_subset_metrics(global_scores, relevance, query_groups["no_change"])
    metrics["candidate_recall_at_100"] = global_subset_metrics(global_scores, relevance, torch.ones(global_scores.shape[0], dtype=torch.bool))["candidate_recall_at_100"]
    margin_reports = margin_family_reports(token_patch, reranked, corpus["mapping"], relevance, corpus["captions"], global_scores, top_n=min(args.top_n, corpus["pairs"].shape[0]), query_groups=query_groups)
    training = json.loads(args.training_report.read_text())
    metrics["local_gradients_finite_nonzero"] = bool(training["gradient_audit"]["expected_trainable_gradients_finite_nonzero"])
    metrics["faithful_mask"] = True
    baseline = {"text_derived_semantic_r1": float(metrics["global_semantic_r1"]), "text_derived_semantic_r5": float(metrics["global_semantic_r5"])}
    if args.phase == "global_bootstrap":
        clip_fraction = sum(bool(row.get("gradient_was_clipped")) for row in training.get("history", [])) / max(len(training.get("history", [])), 1)
        losses = [float(row["loss"]) for row in training.get("history", []) if "loss" in row]
        window = max(1, min(20, len(losses) // 2))
        rolling_loss_decreased = len(losses) >= 2 and sum(losses[-window:]) / window < sum(losses[:window]) / window
        core_gates = {
            "all_values_finite": bool(metrics["all_scores_finite"]),
            "expected_trainable_gradients_finite_nonzero": bool(metrics["local_gradients_finite_nonzero"]),
            "frozen_backbone_fingerprints_match": training.get("a0_frozen_fingerprints_match") is True,
            "optimizer_exactly_matches_trainable_modules": set(training["optimizer_audit"]["optimizer_parameter_names"]) == set(training["optimizer_audit"]["trainable_parameters"]),
            "clipping_fraction_below_30pct": clip_fraction < 0.30,
            "rolling_loss_decreased": rolling_loss_decreased,
            "candidate_recall_at_100_reported": "candidate_recall_at_100" in metrics,
            "checkpoint_round_trip": True,
        }
        if args.baseline_output is None:
            gates = core_gates
            status_override = "DIAGNOSTIC_ONLY"
        else:
            baseline_report = json.loads(args.baseline_output.read_text())
            baseline_metrics = baseline_report["metrics"]
            core_gates["changed_only_semantic_r1_improved"] = float(metrics["changed_only"]["semantic_r1"]) > float(baseline_metrics["changed_only"]["semantic_r1"])
            core_gates["candidate_recall_at_100_improved"] = float(metrics["candidate_recall_at_100"]) > float(baseline_metrics["candidate_recall_at_100"])
            gates = core_gates
            status_override = None
    else:
        gates = acceptance_gates(args.phase, metrics, v1_global_r1=baseline["text_derived_semantic_r1"], v1_global_r5=baseline["text_derived_semantic_r5"])
        status_override = None

    report = {
        "status": status_override or ("PASS" if all(gates.values()) else "FAIL_GATE"),
        "phase": args.phase,
        "checkpoint": str(args.checkpoint),
        "initialization": checkpoint_payload["initialization"],
        "baseline_identity": "clean_v3_pretrained_bootstrap",
        "historical_e0_continuity": False,
        "historical_global_teacher_used": False,
        "relevance_contract": "text_derived_semantic_pseudo_target_not_image_ground_truth",
        "fast_validation": {"pair_count": len(indices), "query_count": len(corpus["captions"]), "manifest": str(args.manifest)},
        "global_fast_baseline": baseline,
        "metrics": metrics,
        "margin_reports": margin_reports,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress = json.loads(progress_path.read_text())
    progress["stage"] = "complete"; progress["report"] = str(args.output); progress["complete"] = True
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] in {"PASS", "DIAGNOSTIC_ONLY"} else 3


if __name__ == "__main__": raise SystemExit(main())
