from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qcpr_single_pass_runtime import (
    collect_gallery,
    collect_queries,
    exact_metrics,
    forbidden_mask_keys,
    git_sha,
    save_checkpoint,
    sha256,
)
from land_change_detection.models.qcpr_single_pass import build_pair_masks, multi_positive_sigmoid_loss
from land_change_detection.models.qcpr_single_pass_factory import build_single_pass_retriever
from land_change_detection.training.qcpr_retrieval_repair import (
    DeterministicRotatingCaptionCollator,
    LogicalBatchContract,
    aggregate_hard_pairs,
    exact_grad_cache_backward,
    hard_aware_epoch_order,
    mine_hard_negative_rows,
    save_hard_negative_cache,
)
from land_change_detection.training.qcpr_single_pass_data import (
    CaptionCollisionIndex,
    MaskFreePairDataset,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--identifiability-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--universat-source", default="/mnt/weka/svardanyan/rs_change_project/external/UniverSat")
    parser.add_argument("--universat-checkpoint", default="/mnt/weka/svardanyan/rs_change_project/models/universat-base")
    parser.add_argument("--jina-model", default="/mnt/weka/svardanyan/rs_change_project/models/jina-v5-text-small-retrieval")
    parser.add_argument("--output-grid", type=int, default=32)
    parser.add_argument("--physical-micro-batch", type=int, default=16)
    parser.add_argument("--logical-physical-batch", type=int, default=128)
    parser.add_argument("--captions-per-pair", type=int, default=2)
    parser.add_argument("--hard-candidates", type=int, default=32)
    parser.add_argument("--hard-warmup-epochs", type=int, default=3)
    parser.add_argument("--hard-refresh-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--recall-regression-tolerance", type=float, default=0.002)
    parser.add_argument("--local-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--disable-hard-mining", action="store_true")
    return parser.parse_args()


def frozen_fingerprint(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1)
        sample = torch.cat((flat[: min(128, flat.numel())], flat[-min(128, flat.numel()) :])).float().cpu()
        digest.update(name.encode())
        digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


def no_decay(name, parameter):
    lowered = name.casefold()
    return parameter.ndim <= 1 or lowered.endswith("bias") or "norm" in lowered or "layer_scale" in lowered or "logit_" in lowered


def optimizer_groups(model, weight_decay):
    groups = []
    covered = set()

    def add(label, named, lr):
        decay, nodecay = [], []
        for name, parameter in named:
            if not parameter.requires_grad or id(parameter) in covered:
                continue
            covered.add(id(parameter))
            (nodecay if no_decay(name, parameter) else decay).append(parameter)
        if decay:
            groups.append({"name": label + "/decay", "params": decay, "lr": lr, "weight_decay": weight_decay})
        if nodecay:
            groups.append({"name": label + "/no_decay", "params": nodecay, "lr": lr, "weight_decay": 0.0})

    add("dense_token_adapters", model.pair_encoder.token_adapters.named_parameters(), 5e-5)
    add("pair_cross_attention", model.pair_encoder.pair_blocks.named_parameters(), 5e-5)
    pair_output = []
    for name in ("pair_token", "baseline_norm", "baseline_projection", "delta_norm", "delta_projection"):
        value = getattr(model.pair_encoder, name)
        pair_output.extend([(name, value)] if isinstance(value, torch.nn.Parameter) else [(name + "." + sub, p) for sub, p in value.named_parameters()])
    add("pair_output", pair_output, 1e-4)
    text = list(model.text_projection.named_parameters())
    local = getattr(model.text_encoder, "local_projection", None)
    if local is not None:
        text.extend(("jina_local." + name, parameter) for name, parameter in local.named_parameters())
    add("text_adapter_projection", text, 1e-4)
    add("logits", [("logit_scale", model.logit_scale), ("logit_bias", model.logit_bias)], 5e-5)
    intended = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if covered != intended:
        raise RuntimeError(f"optimizer coverage mismatch missing={len(intended-covered)} extra={len(covered-intended)}")
    return groups


def extended_evaluation(model, dataset, collisions, batch_size, workers, device):
    gallery = collect_gallery(model, dataset, batch_size, workers, device)
    queries = collect_queries(model, dataset, batch_size, workers, device)
    scores = queries["vectors"] @ gallery["vectors"].T
    metrics, ranks, top10 = exact_metrics(scores, queries["pair_ids"], gallery["pair_ids"], gallery["change_status"])
    metrics["cumulative_recall"] = {str(k): float((ranks <= k).float().mean()) for k in range(1, 101)}
    pair_to_dataset = {
        str(row["pair_id"]): str(row["dataset_name"]) for row in dataset.samples
    }
    order = scores.argsort(dim=1, descending=True)
    gallery_index = {pair_id: index for index, pair_id in enumerate(gallery["pair_ids"])}
    ambiguity_ranks = []
    for row, (pair_id, normalized) in enumerate(zip(queries["pair_ids"], queries["normalized"], strict=True)):
        valid = {pair_id} | collisions.collisions(pair_id, normalized)
        columns = torch.tensor([gallery_index[p] for p in valid if p in gallery_index])
        positions = torch.isin(order[row], columns).nonzero()
        ambiguity_ranks.append(int(positions[0, 0]) + 1)
    ambiguity_ranks = torch.tensor(ambiguity_ranks)
    metrics["ambiguity_aware_exact_duplicate"] = {
        "mrr": float((1 / ambiguity_ranks.float()).mean()),
        "recall_at_1": float((ambiguity_ranks <= 1).float().mean()),
        "recall_at_5": float((ambiguity_ranks <= 5).float().mean()),
        "recall_at_10": float((ambiguity_ranks <= 10).float().mean()),
        "verified_semantic_equivalent_groups": 0,
    }
    metrics["per_dataset"] = {}
    for dataset_name in sorted(set(pair_to_dataset.values())):
        mask = torch.tensor([pair_to_dataset[pair_id] == dataset_name for pair_id in queries["pair_ids"]])
        subset = ranks[mask]
        metrics["per_dataset"][dataset_name] = {
            "queries": int(subset.numel()),
            "mrr": float((1 / subset.float()).mean()),
            "recall_at_1": float((subset <= 1).float().mean()),
            "recall_at_5": float((subset <= 5).float().mean()),
            "recall_at_10": float((subset <= 10).float().mean()),
            "mean_rank": float(subset.float().mean()),
            "median_rank": float(subset.float().median()),
        }
    rows = []
    for row in range(len(queries["captions"])):
        rows.append({
            "query_id": f"query:{row}", "query": queries["captions"][row],
            "normalized_query": queries["normalized"][row], "true_pair_id": queries["pair_ids"][row],
            "dataset": pair_to_dataset[queries["pair_ids"][row]], "true_pair_rank": int(ranks[row]),
            "top10_pair_ids": [gallery["pair_ids"][i] for i in top10[row]],
            "top10_scores": [float(scores[row, i]) for i in top10[row]],
        })
    return metrics, rows


def mine_cache(model, dataset, collisions, output_path, top_k, workers, device, refresh_epoch):
    model.eval()
    gallery = collect_gallery(model, dataset, 16, workers, device)
    queries = collect_queries(model, dataset, 16, workers, device)
    excluded = [collisions.collisions(pair_id, caption) for pair_id, caption in zip(queries["pair_ids"], queries["normalized"], strict=True)]
    rows = mine_hard_negative_rows(
        query_vectors=queries["vectors"], pair_vectors=gallery["vectors"],
        query_pair_ids=queries["pair_ids"], gallery_pair_ids=gallery["pair_ids"],
        query_captions=queries["captions"], excluded_pair_ids=excluded, top_k=top_k,
    )
    path = output_path / f"hard_negatives_epoch_{refresh_epoch:02d}.jsonl"
    save_hard_negative_cache(path, rows)
    stats = {
        "refresh_epoch": refresh_epoch, "queries": len(rows), "top_k": top_k,
        "mean_excluded_pairs": sum(row["excluded_pair_count"] for row in rows) / max(len(rows), 1),
        "mean_hard_score": sum(sum(row["hard_scores"]) for row in rows) / max(len(rows) * top_k, 1),
        "cache_path": str(path), "cache_sha256": sha256(path),
        "selection": "model scores only", "handwritten_categories": False,
    }
    return rows, aggregate_hard_pairs(rows), stats


def main():
    args = arguments()
    if args.local_loss_weight != 0:
        raise ValueError("R1 requires local_loss_weight=0; the implemented FILIP head is reserved for controlled R3")
    contract = LogicalBatchContract(args.physical_micro_batch, args.logical_physical_batch, args.captions_per_pair)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reproduction = json.loads(args.reproduction.read_text())
    if not reproduction.get("passed"):
        raise RuntimeError("immutable baseline reproduction did not pass")
    if not args.identifiability_audit.is_file():
        raise RuntimeError("identifiability audit is missing")
    train_manifest = args.manifest_dir / "natural_train_retrieval_manifest.jsonl"
    val_manifest = args.manifest_dir / "natural_validation_retrieval_manifest.jsonl"
    collision_path = args.manifest_dir / "caption_quality_audit.jsonl"
    train = MaskFreePairDataset(train_manifest, "train")
    val = MaskFreePairDataset(val_manifest, "val")
    collisions = CaptionCollisionIndex(collision_path)
    model = build_single_pass_retriever(
        universat_source=args.universat_source, universat_checkpoint=args.universat_checkpoint,
        jina_model=args.jina_model, device=device, output_grid=args.output_grid,
    )
    baseline_payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    if baseline_payload.get("role") != "retrieval" or baseline_payload.get("epoch") != 19:
        raise RuntimeError("R1 must initialize from accepted epoch-19 retrieval model")
    if sha256(args.baseline_checkpoint) != reproduction["checkpoint_sha256"]:
        raise RuntimeError("baseline checkpoint SHA mismatch")
    model.load_state_dict(baseline_payload["model"])
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay))
    logical_batches_per_epoch = math.ceil(len(train) / args.logical_physical_batch)
    total_steps = args.max_steps if args.max_steps > 0 else logical_batches_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    def schedule(step):
        if step < warmup:
            return max(step, 1) / warmup
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    baseline_metrics = reproduction["reproduced_metrics"]
    config = vars(args) | {
        "git_sha": git_sha(), "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
        "fresh_optimizer": True, "fresh_scheduler": True, "old_optimizer_loaded": False,
        "train_manifest_sha256": sha256(train_manifest), "validation_manifest_sha256": sha256(val_manifest),
        "collision_audit_sha256": sha256(collision_path), "logical_batch_contract": asdict(contract),
        "logical_score_matrix": list(contract.score_matrix_shape), "gradient_accumulation_is_logical_batch": False,
        "local_token_patch_weight": 0.0, "mask_supervision": False,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
    visual_before = frozen_fingerprint(model.visual_encoder)
    text_before = frozen_fingerprint(model.text_encoder)
    hard_by_pair = None
    hard_history = []
    actual_micro = args.physical_micro_batch
    best = None
    best_metrics = None
    stale = 0
    global_step = 0
    pair_presentations = Counter()
    query_presentations = Counter()
    source_presentations = Counter()
    change_presentations = Counter()
    schedule_hasher = hashlib.sha256()
    history = args.output_dir / "metrics.jsonl"
    first_batch_checked = False
    for epoch in range(args.epochs):
        if (not args.disable_hard_mining and epoch >= args.hard_warmup_epochs and
                (epoch - args.hard_warmup_epochs) % args.hard_refresh_epochs == 0):
            rows, hard_by_pair, mining_stats = mine_cache(model, train, collisions, args.output_dir, args.hard_candidates, args.workers, device, epoch)
            hard_history.append(mining_stats)
            (args.output_dir / "hard_negative_manifest.json").write_text(json.dumps(hard_history, indent=2, sort_keys=True) + "\n")
        order, ordering_stats = hard_aware_epoch_order(
            train.pair_ids, logical_batch=args.logical_physical_batch, seed=args.seed + epoch,
            hard_by_pair=hard_by_pair,
        )
        loader = DataLoader(
            train, batch_size=actual_micro, sampler=order, num_workers=args.workers, pin_memory=True,
            collate_fn=DeterministicRotatingCaptionCollator(args.captions_per_pair, args.seed, epoch), drop_last=False,
        )
        iterator = iter(loader)
        model.train()
        sums = defaultdict(float)
        logical_count = 0
        while logical_count < ordering_stats["logical_batches"]:
            micro_batches = [next(iterator) for _ in range(contract.micro_batches_per_logical_batch)]
            if any(forbidden_mask_keys(batch) for batch in micro_batches):
                raise RuntimeError("segmentation target entered R1 retrieval")
            query_ids = sum((batch["query_pair_ids"] for batch in micro_batches), [])
            normalized = sum((batch["normalized_captions"] for batch in micro_batches), [])
            pair_ids = sum((batch["pair_ids"] for batch in micro_batches), [])
            for pair_id in pair_ids:
                pair_presentations[str(pair_id)] += 1
                sample = train.samples[train.pair_ids.index(str(pair_id))]
                source_presentations[str(sample.get("dataset_name", "unknown"))] += 1
                change_presentations["no_change" if sample.get("source_metadata", {}).get("changeflag") == 0 else "changed"] += 1
            for pair_id, caption in zip(query_ids, normalized, strict=True):
                query_presentations[f"{pair_id}:{caption}"] += 1
            schedule_hasher.update(json.dumps({"epoch": epoch + 1, "step": global_step + 1, "pairs": [str(x) for x in pair_ids], "queries": [f"{p}:{c}" for p, c in zip(query_ids, normalized, strict=True)]}, sort_keys=True).encode())
            collision_sets = [collisions.collisions(pair_id, caption) for pair_id, caption in zip(query_ids, normalized, strict=True)]
            positive, excluded = build_pair_masks(query_ids, pair_ids, collision_sets, device)
            def encode(batch):
                output = model(batch["images"].to(device, non_blocking=True), batch["captions"])
                return output.text.text_search_vector, output.pair.pair_search_vector
            def loss_function(text, pair):
                logits = model.scale * (text @ pair.T) + model.logit_bias
                return multi_positive_sigmoid_loss(logits, positive, excluded)
            optimizer.zero_grad(set_to_none=True)
            if not first_batch_checked:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            loss, stats, shape = exact_grad_cache_backward(
                micro_batches=micro_batches, encode=encode, loss_function=loss_function, device=device,
                autocast_factory=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
            )
            if shape != contract.score_matrix_shape:
                raise RuntimeError(f"logical contrastive matrix mismatch: {shape}")
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("non-finite R1 loss or gradients")
            if not first_batch_checked:
                peak_allocated = torch.cuda.max_memory_allocated() / 2**30
                peak_reserved = torch.cuda.max_memory_reserved() / 2**30
                if frozen_fingerprint(model.visual_encoder) != visual_before or frozen_fingerprint(model.text_encoder) != text_before:
                    raise RuntimeError("frozen backbone fingerprint changed")
                first_batch_checked = True
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            logical_count += 1
            sums["loss"] += float(loss)
            for key, value in stats.items():
                sums[key] += float(value)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        model.eval()
        metrics, rows = extended_evaluation(model, val, collisions, 16, args.workers, device)
        all_metrics = metrics["all"]
        feasible = all_metrics["recall_at_10"] + args.recall_regression_tolerance >= baseline_metrics["all"]["recall_at_10"]
        selector = (all_metrics["mrr"], -all_metrics["median_rank"], all_metrics["recall_at_10"])
        record = {
            "epoch": epoch + 1, "global_step": global_step,
            **{key: value / max(logical_count, 1) for key, value in sums.items()},
            "development": metrics, "selector": selector, "recall10_feasible": feasible,
            "ordering": ordering_stats, "hard_negative_refreshes": len(hard_history),
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
        }
        with history.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        extra = {
            "baseline_checkpoint": str(args.baseline_checkpoint), "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
            "baseline_epoch": 19, "fresh_optimizer": True, "logical_batch_contract": asdict(contract),
            "logical_score_matrix": list(contract.score_matrix_shape), "hard_negative_history": hard_history,
            "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
        }
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch + 1, metrics, config, "retrieval_repair_r1", scheduler=scheduler, global_step=global_step, extra=extra)
        if feasible and (best is None or selector > best):
            best, best_metrics, stale = selector, metrics, 0
            save_checkpoint(args.output_dir / "best_r1.pt", model, optimizer, epoch + 1, metrics, config, "retrieval_repair_r1", scheduler=scheduler, global_step=global_step, extra=extra)
            with (args.output_dir / "best_top10.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        else:
            stale += 1
        if args.max_steps <= 0 and epoch + 1 >= args.min_epochs and stale >= args.patience:
            break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if args.max_steps > 0 and global_step != args.max_steps:
        raise RuntimeError(f"fixed-step contract violated: global_step={global_step}, requested={args.max_steps}")
    if best_metrics is None:
        raise RuntimeError("R1 produced no Recall@10-feasible checkpoint")
    accepted = best_metrics["all"]
    meaningful_gate = {
        "mrr_at_least_0_030": accepted["mrr"] >= 0.030,
        "recall_at_10_at_least_0_060": accepted["recall_at_10"] >= 0.060,
        "median_rank_at_most_150": accepted["median_rank"] <= 150,
    }
    acceptance = {
        "process_completed": True,
        "promoted_automatically": False,
        "meaningful_repair_gate_passed": all(meaningful_gate.values()),
        "meaningful_repair_gate": meaningful_gate,
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "progress": {key: accepted[key] - baseline_metrics["all"][key] for key in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "mean_rank", "median_rank")},
        "checkpoint_path": str(args.output_dir / "best_r1.pt"),
        "checkpoint_sha256": sha256(args.output_dir / "best_r1.pt"),
        "recommended_next_stage": "manual scientific review before R2",
    }
    (args.output_dir / "r1_acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    batch_contract = {
        "gpu_name": torch.cuda.get_device_name(device), "physical_micro_batch": actual_micro,
        "logical_physical_batch": args.logical_physical_batch, "captions_per_pair": args.captions_per_pair,
        "logical_text_queries": contract.logical_query_count, "logical_score_matrix": list(contract.score_matrix_shape),
        "micro_batches_per_logical_batch": contract.micro_batches_per_logical_batch,
        "gradient_accumulation": 1, "gradcache_recomputation": True,
        "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
    }
    (args.output_dir / "batch_contract.json").write_text(json.dumps(batch_contract, indent=2, sort_keys=True) + "\n")
    exposure = {
        "total_pair_presentations": int(sum(pair_presentations.values())),
        "total_query_presentations": int(sum(query_presentations.values())),
        "unique_pairs": len(pair_presentations),
        "unique_queries": len(query_presentations),
        "presentations_per_pair": dict(sorted(pair_presentations.items())),
        "presentations_per_query": dict(sorted(query_presentations.items())),
        "mean_presentations_per_pair": float(sum(pair_presentations.values()) / max(len(pair_presentations), 1)),
        "mean_presentations_per_query": float(sum(query_presentations.values()) / max(len(query_presentations), 1)),
        "per_source_presentations": dict(sorted(source_presentations.items())),
        "changed_no_change_presentations": dict(sorted(change_presentations.items())),
        "sampling_schedule_sha256": schedule_hasher.hexdigest(),
        "global_steps": global_step,
        "logical_batch_contract": asdict(contract),
    }
    (args.output_dir / "exposure_accounting.json").write_text(json.dumps(exposure, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "training_complete.json").write_text(json.dumps({"epochs": epoch + 1, "global_step": global_step, "best_selector": best, "batch_contract": batch_contract}, indent=2) + "\n")


if __name__ == "__main__":
    main()
