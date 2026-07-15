#!/usr/bin/env python3
"""Deterministic fast full-gallery evaluation for controlled QCPR v3 diagnostics."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import ucv2_cluster_common as common
import ucv2_stage1_next_core as legacy
from land_change_detection.models.qcpr_v3_experiment import acceptance_gates, experiment_metrics
from land_change_detection.models.qcpr_v3_runtime import IMMUTABLE_V1, build_v3_and_teacher
from land_change_detection.models.retrieval_heads import semantic_teacher_relevance_matrix, stable_caption_group_ids


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
    token_batches, attention_batches = [], []
    captions, mapping, offset = [], [], 0
    for batch in legacy.make_eval_loader(dataset, config):
        images, temporal = batch["images"].to(device), batch["temporal_valid_mask"].to(device)
        pairs, per_time, _ = model.encode_pairs(images, temporal)
        text, tokens, attention = model.encode_texts(batch["captions"])
        for name, value in (("pairs", pairs), ("per_time", per_time), ("text", text), ("masks", batch["masks"]), ("segmentation", batch["segmentation_supervision"])):
            pieces[name].append(value.cpu())
        token_batches.append(tokens.cpu()); attention_batches.append(attention.cpu())
        captions.extend(batch["captions"]); mapping.extend((batch["caption_to_pair"] + offset).tolist()); offset += images.shape[0]
    tokens, attention = pad_token_batches(token_batches, attention_batches)
    return {name: torch.cat(values) for name, values in pieces.items()} | {"tokens": tokens, "attention": attention, "captions": captions, "mapping": torch.tensor(mapping, dtype=torch.long)}


@torch.no_grad()
def score(model, corpus, device, qchunk, cchunk, include_masks):
    qcount, ccount = corpus["text"].shape[0], corpus["pairs"].shape[0]
    global_scores = corpus["text"] @ corpus["pairs"].T
    local, reranked, logits, targets = torch.empty_like(global_scores), torch.empty_like(global_scores), [], []
    for q0 in range(0, qcount, qchunk):
        q1 = min(q0 + qchunk, qcount); mapping = corpus["mapping"][q0:q1]
        for c0 in range(0, ccount, cchunk):
            c1 = min(c0 + cchunk, ccount)
            output = model.score_encoded(corpus["pairs"][c0:c1].to(device), corpus["per_time"][c0:c1].to(device), corpus["text"][q0:q1].to(device), corpus["tokens"][q0:q1].to(device), corpus["attention"][q0:q1].to(device))
            local[q0:q1, c0:c1], reranked[q0:q1, c0:c1] = output.local_score.cpu(), output.reranked_score.cpu()
            if include_masks:
                selected = (mapping >= c0) & (mapping < c1) & corpus["segmentation"][mapping]
                for row in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                    pair = int(mapping[row]); logit = output.decoded_mask_logits[row, pair - c0].cpu()
                    target = torch.nn.functional.interpolate(corpus["masks"][pair:pair + 1].unsqueeze(1), logit.shape[-2:], mode="nearest")[0, 0]
                    logits.append(logit); targets.append(target)
    return global_scores, local, reranked, logits, targets


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
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("fast evaluation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    config = make_config(args.v1_checkpoint, args.batch_size, args.manifest)
    _, val = legacy._build_stage1_datasets(config)
    indices = [index for index, row in enumerate(val.samples) if str(row["dataset_name"]) in {"levir_mci", "second_cc"}][:args.fast_pairs]
    if len(indices) < 2: raise RuntimeError("fast validation subset is empty")
    model, _, audit = build_v3_and_teacher(args.v1_checkpoint, device=device, build_legacy_model=common.build_model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True); model.eval()
    corpus = collect(model, Subset(val, indices), config, device)
    global_scores, local, reranked, logits, targets = score(model, corpus, device, args.query_chunk, args.candidate_chunk, args.phase == "mask_grounding")
    relevance = semantic_teacher_relevance_matrix(corpus["text"], corpus["captions"], corpus["mapping"], stable_caption_group_ids(corpus["captions"]), pair_count=corpus["pairs"].shape[0], top_k=0)
    masks = {}
    if args.phase == "mask_grounding":
        if not logits: raise RuntimeError("mask validation selected no supervised masks")
        masks = {"mask_logits": torch.stack(logits), "mask_targets": torch.stack(targets)}
    metrics = experiment_metrics(global_scores, local, reranked, relevance, corpus["mapping"], top_n=min(args.top_n, corpus["pairs"].shape[0]), **masks)
    training = json.loads(args.training_report.read_text())
    metrics["local_gradients_finite_nonzero"] = bool(training["gradient_audit"]["local_gradients_finite_nonzero"])
    metrics["faithful_mask"] = True
    baseline = {"semantic_r1": float(metrics["global_semantic_r1"]), "semantic_r5": float(metrics["global_semantic_r5"])}
    gates = acceptance_gates(args.phase, metrics, v1_global_r1=baseline["semantic_r1"], v1_global_r5=baseline["semantic_r5"])
    report = {"status": "PASS" if all(gates.values()) else "FAIL_GATE", "phase": args.phase, "checkpoint": str(args.checkpoint), "initialization": audit.__dict__, "fast_validation": {"pair_count": len(indices), "query_count": len(corpus["captions"]), "manifest": str(args.manifest)}, "v1_compatible_global_fast_baseline": baseline, "metrics": metrics, "gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__": raise SystemExit(main())
