#!/usr/bin/env python3
"""Bounded model-only smoke using schema-shaped synthetic token inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qcpr_v3.config import config_dict, default_config, validate_config
from qcpr_v3.data.contracts import TemporalMetadata
from qcpr_v3.diagnostics.evidence import evidence_gradient_diagnostics
from qcpr_v3.evaluation.retrieval import rank_scores, retrieval_metrics
from qcpr_v3.models import QCPRV3Model
from qcpr_v3.training.checkpointing import save_checkpoint
from qcpr_v3.training.exposure import ExposureAccounting, record_step
from qcpr_v3.training.objective import UnifiedListwiseLoss
from qcpr_v3.training.trainer import train_step


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 32:
        raise SystemExit("smoke steps must be in [1,32]")
    random.seed(20260804)
    np.random.seed(20260804)
    torch.manual_seed(20260804)
    device = torch.device(args.device)
    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.config_path is None:
        config = default_config()
    else:
        payload = json.loads(args.config_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("config JSON must be an object")
        config = validate_config(payload)
    model = QCPRV3Model(config).to(device)
    model.train()
    objective = UnifiedListwiseLoss(temperature=config.training.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    pair_count, query_count, time_count, token_count, length = 4, 4, 2, 16, 8
    native_tokens = torch.randn(pair_count, time_count, token_count, config.temporal.native_dim, device=device)
    coordinates = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]] * 4,
        dtype=torch.float32,
        device=device,
    ).reshape(1, token_count, 2).expand(pair_count, -1, -1)
    timestamps = torch.arange(time_count, device=device, dtype=torch.float32).view(1, -1).expand(pair_count, -1)
    metadata = TemporalMetadata(
        timestamps=timestamps,
        delta_times=timestamps - timestamps[:, :1],
        frame_ids=torch.arange(time_count, device=device).view(1, -1).expand(pair_count, -1),
    )
    text_tokens = torch.randn(query_count, length, config.text.input_dim, device=device)
    text_mask = torch.ones(query_count, length, dtype=torch.bool, device=device)
    grades = torch.eye(query_count, pair_count, device=device, dtype=torch.long)
    accounting = ExposureAccounting([], [], [], [])
    metrics_history: list[dict[str, Any]] = []
    for step in range(args.steps):
        pair_ids = [f"pair-{index}" for index in range(pair_count)]
        query_ids = [f"query-{index}" for index in range(query_count)]
        record_step(accounting, pair_ids, query_ids)
        # Rebuild the trainable adapter graph every step. Reusing encoded
        # outputs would attempt a second backward through a freed graph.
        visual = model.encode_visual(native_tokens, coordinates, metadata)
        query = model.encode_query(text_tokens, text_mask)
        result = train_step(model, query, visual, grades, objective=objective, optimizer=optimizer)
        if not torch.isfinite(result.scores).all() or not torch.isfinite(result.loss.loss):
            raise FloatingPointError("non-finite smoke score or loss")
        metrics_history.append({"step": step + 1, "loss": float(result.loss.loss.detach()), "evidence_gradient_norm": result.evidence_gradient_norm})

    model.eval()
    with torch.no_grad():
        visual = model.encode_visual(native_tokens, coordinates, metadata)
        query = model.encode_query(text_tokens, text_mask)
        final = model.score(query, visual)
        metrics = retrieval_metrics(final.scores, grades.bool(), ks=(1, 2, 4))
        ranking_order = rank_scores(final.scores).cpu()
    checkpoint = args.run_root / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        metadata={"code_sha": args.expected_sha, "data_release": args.data_release, "status": "SMOKE"},
    )
    checkpoint_digest = file_sha256(checkpoint)
    full_rankings = args.run_root / "full_rankings.pt"
    torch.save(final.scores.detach().cpu(), full_rankings)
    rankings_top100 = args.run_root / "rankings_top100.jsonl"
    with rankings_top100.open("w") as handle:
        for query_index, ordered in enumerate(ranking_order.tolist()):
            handle.write(json.dumps({"query_id": f"query-{query_index}", "ranked_item_ids": [f"pair-{index}" for index in ordered[:100]]}) + "\n")
    ranking_digest = file_sha256(full_rankings)
    (args.run_root / "checkpoint.sha256").write_text(checkpoint_digest + "\n")
    (args.run_root / "code_state.json").write_text(json.dumps({"expected_sha": args.expected_sha, "python": os.sys.executable, "status": "SMOKE"}, indent=2) + "\n")
    (args.run_root / "config_resolved.json").write_text(json.dumps({"config": config_dict(config), "steps": args.steps}, indent=2, sort_keys=True) + "\n")
    (args.run_root / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, indent=2) + "\n")
    (args.run_root / "model_contract.json").write_text(json.dumps({"architecture_id": config.architecture_id, "sequence_cls": [pair_count, 512], "dense_tokens": [pair_count, time_count, token_count, 512], "evidence_map": [query_count, pair_count, time_count, token_count], "parameter_counts": model.parameter_counts()}, indent=2) + "\n")
    (args.run_root / "parameter_groups.json").write_text(json.dumps({"all": sum(p.numel() for p in model.parameters() if p.requires_grad)}, indent=2) + "\n")
    (args.run_root / "batch_contract.json").write_text(json.dumps({"physical_microbatch": pair_count, "logical_physical_batch": pair_count, "logical_score_matrix": [query_count, pair_count], "mask_access": "none"}, indent=2) + "\n")
    (args.run_root / "exposure_accounting.json").write_text(json.dumps(accounting.as_dict(), indent=2) + "\n")
    (args.run_root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics_history) + "\n")
    (args.run_root / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.run_root / "retrieval_integrity_audit.json").write_text(json.dumps({"ranking_shape": list(final.scores.shape), "ranking_sha256": ranking_digest, "query_ids_sha256": hashlib.sha256("\n".join(f"query-{i}" for i in range(query_count)).encode()).hexdigest(), "gallery_ids_sha256": hashlib.sha256("\n".join(f"pair-{i}" for i in range(pair_count)).encode()).hexdigest()}, indent=2) + "\n")
    (args.run_root / "embedding_diagnostics.json").write_text(json.dumps({"sequence_cls_norm_mean": float(visual.sequence_cls.norm(dim=-1).mean().detach()), "text_cls_norm_mean": float(query.text_cls.norm(dim=-1).mean().detach())}, indent=2) + "\n")
    (args.run_root / "gradient_diagnostics.json").write_text(json.dumps(evidence_gradient_diagnostics(model), indent=2) + "\n")
    (args.run_root / "evidence_diagnostics.json").write_text(json.dumps({"weights_shape": list(final.evidence_weights.shape) if final.evidence_weights is not None else None, "nonzero": bool(final.evidence_weights is not None and (final.evidence_weights > 0).any())}, indent=2) + "\n")
    (args.run_root / "localization_metrics.json").write_text(json.dumps({"mask_access": "evaluation_only", "diagnostics": "not_supervised_in_smoke"}, indent=2) + "\n")
    (args.run_root / "training_complete.json").write_text(json.dumps({"status": "COMPLETED_FIXED_STEPS", "global_step": args.steps, "requested_steps": args.steps, "checkpoint_sha256": checkpoint_digest}, indent=2) + "\n")
    sums = {str(path.name): file_sha256(path) for path in args.run_root.iterdir() if path.is_file() and path.name != "SHA256SUMS"}
    (args.run_root / "SHA256SUMS").write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(sums.items())) + "\n")


if __name__ == "__main__":
    main()
