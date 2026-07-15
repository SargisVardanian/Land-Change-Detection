#!/usr/bin/env python3
"""Immutable full-gallery forensic evaluation for the historical QCPR E0 run."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import ucv2_cluster_common as common
import ucv2_stage1_next_core as legacy
import ucv2_retrieval_metrics as metrics
from land_change_detection.models.retrieval_heads import classify_caption_semantics, semantic_teacher_relevance_matrix


def semantic_summary(scores: torch.Tensor, relevance: torch.Tensor, keep: torch.Tensor) -> dict:
    keep = keep.bool()
    if not bool(keep.any()):
        return {"query_count": 0, "semantic_r1": 0.0, "semantic_r5": 0.0, "semantic_r10": 0.0, "semantic_ndcg10": 0.0}
    scores, relevance = scores[keep], relevance[keep].bool()
    order = scores.argsort(1, descending=True, stable=True)
    ranked = relevance.gather(1, order)
    discounts = 1.0 / torch.log2(torch.arange(scores.shape[1], dtype=torch.float32) + 2.0)
    gains = ranked[:, :10].float()
    dcg = (gains * discounts[:gains.shape[1]]).sum(1)
    ideal = torch.sort(relevance.float(), 1, descending=True).values[:, :10]
    idcg = (ideal * discounts[:ideal.shape[1]]).sum(1).clamp_min(1e-8)
    return {
        "query_count": int(keep.sum()),
        "semantic_r1": float(ranked[:, :1].any(1).float().mean()),
        "semantic_r5": float(ranked[:, :5].any(1).float().mean()),
        "semantic_r10": float(ranked[:, :10].any(1).float().mean()),
        "semantic_ndcg10": float((dcg / idcg).mean()),
    }


def candidate_recall(scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
    top = scores.topk(min(k, scores.shape[1]), dim=1).indices
    return float(relevance.bool().gather(1, top).any(1).float().mean())


def margins(scores: torch.Tensor, positive: torch.Tensor) -> dict:
    positive = positive.bool()
    best_positive = scores.masked_fill(~positive, float("-inf")).amax(1)
    best_negative = scores.masked_fill(positive, float("-inf")).amax(1)
    values = best_positive - best_negative
    valid = torch.isfinite(values)
    values = values[valid]
    return {
        "query_count": int(values.numel()), "mean": float(values.mean()) if values.numel() else 0.0,
        "median": float(values.median()) if values.numel() else 0.0,
        "p10": float(torch.quantile(values, .1)) if values.numel() else 0.0,
        "minimum": float(values.min()) if values.numel() else 0.0,
        "fraction_le_zero": float((values <= 0).float().mean()) if values.numel() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full E0 forensic evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    values.update(batch_size=args.batch_size, num_workers=args.num_workers, max_val_samples=None)
    fields = legacy.Stage1NextConfig.__dataclass_fields__
    config = legacy.Stage1NextConfig(**{key: value for key, value in values.items() if key in fields})
    _, val = legacy._build_stage1_datasets(config)
    device = torch.device("cuda", torch.cuda.current_device())
    model = common.build_model(config, device)
    model.load_state_dict(payload["model"], strict=True)
    corpus = metrics.collect_retrieval_corpus(model, legacy.make_eval_loader(val, config), device, config)
    rank = metrics.compute_retrieval_ranks(corpus, query_chunk_size=64, candidate_chunk_size=256)
    core, global_scores = metrics.compute_retrieval_metrics(corpus, rank_result=rank)
    teacher = corpus.teacher_text_embeddings if corpus.teacher_text_embeddings is not None else corpus.text_embeddings
    relevance = semantic_teacher_relevance_matrix(teacher, corpus.captions, corpus.caption_to_pair, corpus.caption_group_ids, pair_count=len(corpus.pair_ids), top_k=0) > 0
    duplicate_positive = rank.positive_mask
    exact_positive = torch.nn.functional.one_hot(corpus.caption_to_pair, len(corpus.pair_ids)).bool()
    semantics = [classify_caption_semantics(caption) for caption in corpus.captions]
    changed = torch.tensor([bool(item["changed"]) for item in semantics])
    no_change = torch.tensor([bool(item["no_change"]) for item in semantics])
    cluster_sizes = Counter(corpus.caption_group_ids.tolist())
    frequent = torch.tensor([cluster_sizes[int(group)] >= 20 for group in corpus.caption_group_ids.tolist()])
    rare = torch.tensor([cluster_sizes[int(group)] <= 5 for group in corpus.caption_group_ids.tolist()])
    # Text-only caption-prototype baseline: candidate representation is the
    # mean of captions attached to it, so it intentionally uses no image data.
    prototypes = torch.zeros((len(corpus.pair_ids), teacher.shape[1]), dtype=torch.float32)
    counts = torch.zeros(len(corpus.pair_ids), dtype=torch.float32)
    prototypes.index_add_(0, corpus.caption_to_pair, teacher.float())
    counts.index_add_(0, corpus.caption_to_pair, torch.ones_like(corpus.caption_to_pair, dtype=torch.float32))
    prototypes = torch.nn.functional.normalize(prototypes / counts.clamp_min(1)[:, None], dim=1)
    text_only_scores = torch.nn.functional.normalize(teacher.float(), dim=1) @ prototypes.T
    pair_no_change = torch.zeros(len(corpus.pair_ids), dtype=torch.bool)
    for qi, pair in enumerate(corpus.caption_to_pair.tolist()):
        pair_no_change[pair] |= no_change[qi]
    # Static frequency baseline: all no-change pairs get priority, ties retain
    # deterministic pair order. It is a diagnostic, not a visual model.
    no_change_frequency_scores = pair_no_change.float().repeat(len(corpus.captions), 1)
    random_scores = -torch.arange(len(corpus.pair_ids), dtype=torch.float32).repeat(len(corpus.captions), 1)
    groups = {"all": torch.ones(len(corpus.captions), dtype=torch.bool), "changed_only": changed, "no_change": no_change, "frequent_caption": frequent, "rare_caption": rare}
    report = {
        "status": "PASS", "kind": "immutable_historical_forensic_evaluation", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "gallery": {"pairs": len(corpus.pair_ids), "queries": len(corpus.captions), "full_natural_validation": True},
        "exact_pair_diagnostic": {key: value for key, value in core.items() if str(key).startswith("exact_pair_")},
        "duplicate_aware_caption_metrics": {key: value for key, value in core.items() if str(key).startswith("text_to_pair") or key in {"MRR", "median_rank", "mean_rank", "positive_count_min", "positive_count_mean", "positive_count_max"}},
        "teacher_semantic_metrics": {key: value for key, value in core.items() if str(key).startswith("semantic_")},
        "candidate_recall": {str(k): candidate_recall(global_scores, relevance, k) for k in (50,100,200)},
        "positive_set_sizes": {"exact_mean": 1.0, "duplicate_aware_mean": float(duplicate_positive.sum(1).float().mean()), "teacher_semantic_mean": float(relevance.sum(1).float().mean()), "teacher_semantic_min": int(relevance.sum(1).min()), "teacher_semantic_max": int(relevance.sum(1).max())},
        "best_positive_minus_best_negative": {"exact_pair": margins(global_scores, exact_positive), "duplicate_aware": margins(global_scores, duplicate_positive), "teacher_semantic": margins(global_scores, relevance)},
        "strata": {name: {"teacher_semantic": semantic_summary(global_scores, relevance, keep), "exact_pair": semantic_summary(global_scores, exact_positive, keep), "duplicate_aware": semantic_summary(global_scores, duplicate_positive, keep), "query_count": int(keep.sum())} for name, keep in groups.items()},
        "baselines": {
            "text_only_caption_prototype_teacher_semantic": semantic_summary(text_only_scores, relevance, groups["all"]),
            "random_deterministic_order_teacher_semantic": semantic_summary(random_scores, relevance, groups["all"]),
            "no_change_frequency_teacher_semantic": semantic_summary(no_change_frequency_scores, relevance, groups["all"]),
        },
        "notes": [
            "exact_pair is diagnostic only; repeated or paraphrased captions can describe several valid pairs.",
            "teacher_semantic relevance is model-derived/pseudo-label evidence, not human-verified structured relevance.",
            "no_change_frequency is a non-visual static baseline; it exposes class-frequency inflation.",
        ],
    }
    torch.save({"global_scores": global_scores, "exact_mapping": corpus.caption_to_pair, "teacher_semantic_relevance": relevance, "duplicate_positive": duplicate_positive, "captions": corpus.captions, "pair_ids": corpus.pair_ids, "dataset_names": corpus.dataset_names}, args.output_dir / "forensic_score_artifact.pt")
    (args.output_dir / "forensic_baseline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
