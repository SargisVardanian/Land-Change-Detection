#!/usr/bin/env python3
"""Audit source separability of query-independent pair embeddings.

This is an evaluation-only diagnostic.  It does not update model weights and
does not read labels, masks, or generated text.  A balanced LEVIR/SECOND
development sample is encoded at each explicit NaFlex patch budget and a
small deterministic linear probe estimates whether the embedding exposes the
source identity.  The result is evidence about a possible shortcut, not a
training metric.
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
from qcpr_siglip2.data.runtime import encode_real_images
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


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


def _load_adapter_checkpoint(
    model: Siglip2TemporalRetrievalModel, checkpoint: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain model_state")
    # Backbone weights are intentionally excluded: the audit evaluates the
    # pinned backbone named on the command line and only restores the trained
    # temporal/evidence/relevance adapter state.
    adapter_state = {
        key: value for key, value in state.items() if not key.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("backbone.")]
    if unexpected:
        raise ValueError(f"unexpected adapter checkpoint keys: {unexpected[:8]}")
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_keys": len(state),
        "restored_adapter_keys": len(adapter_state),
        "missing_backbone_or_optional_keys": list(missing),
    }


def _effective_rank(embeddings: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(embeddings - embeddings.mean(dim=0, keepdim=True))
    mass = singular / singular.sum().clamp_min(1e-12)
    entropy = -(mass * mass.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _probe_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, seed: int) -> dict[str, float]:
    if labels.unique().numel() != 2:
        raise ValueError("source probe requires two source classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(labels.numel(), generator=generator)
    fold_scores: list[float] = []
    for fold in range(5):
        test_indices = order[fold::5]
        train_indices = order[~torch.isin(torch.arange(labels.numel()), test_indices)]
        x_train = embeddings[train_indices].float()
        x_test = embeddings[test_indices].float()
        y_train = labels[train_indices].float()
        y_test = labels[test_indices]
        mean = x_train.mean(dim=0, keepdim=True)
        scale = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        x_train = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
        x_test = torch.cat([x_test, torch.ones(x_test.shape[0], 1)], dim=1)
        regularizer = torch.eye(x_train.shape[1]) * 1e-2
        regularizer[-1, -1] = 0.0
        weights = torch.linalg.solve(
            x_train.T @ x_train + regularizer, x_train.T @ y_train
        )
        prediction = (x_test @ weights >= 0.5).long()
        fold_scores.append(float((prediction == y_test).float().mean()))
    values = torch.tensor(fold_scores)
    return {
        "mean_accuracy": float(values.mean()),
        "std_accuracy": float(values.std(unbiased=False)),
        "fold_accuracy": fold_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--max-pairs-per-source", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 576, 1024])
    parser.add_argument("--seed", type=int, default=20260809)
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
    config = _load_config(Path(args.config_path))
    rows = load_exact_pair_rows(manifest, split="development")
    groups = group_rows_by_pair(rows)
    by_source: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for pair_id, group in groups.items():
        source = str(group[0]["dataset_name"])
        by_source.setdefault(source, []).append((pair_id, group[0]))
    required_sources = ("levir_mci", "second_cc")
    if any(source not in by_source for source in required_sources):
        raise ValueError("development manifest must contain LEVIR-MCI and SECOND-CC")
    count = min(args.max_pairs_per_source, *(len(by_source[source]) for source in required_sources))
    selected: list[dict[str, Any]] = []
    labels: list[int] = []
    for label, source in enumerate(required_sources):
        for pair_id, row in sorted(by_source[source], key=lambda item: item[0])[:count]:
            selected.append(row)
            labels.append(label)
    pair_ids = [str(row["canonical_pair_id"]) for row in selected]
    result: dict[str, Any] = {
        "status": "PASS",
        "evaluation_only": True,
        "source_shortcut_threshold": 0.75,
        "code": state,
        "expected_code_sha": args.expected_code_sha,
        "release": str(release),
        "release_sha256": _sha256(release / "SHA256SUMS") if (release / "SHA256SUMS").is_file() else None,
        "development_manifest": str(manifest),
        "development_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "model": str(args.siglip2_model),
        "config": config.to_dict(),
        "sources": {source: count for source in required_sources},
        "pair_count": len(selected),
        "pair_ids_sha256": _sha256_lines(pair_ids),
        "budgets": {},
    }
    backbone = Siglip2Backbone(
        args.siglip2_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda()
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    model = Siglip2TemporalRetrievalModel(backbone, config).cuda().eval()
    result["checkpoint_restore"] = _load_adapter_checkpoint(model, checkpoint)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    device = torch.device("cuda")
    for budget in args.budgets:
        if budget not in config.supported_patch_budgets:
            raise ValueError(f"unsupported explicit patch budget: {budget}")
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            image = encode_real_images(
                backbone,
                processor,
                selected,
                device,
                dtype=torch.bfloat16,
                no_grad=True,
                max_num_patches=budget,
            )
            temporal = model.temporal_adapter(
                image.patch_tokens,
                image.pooled_embedding,
                patch_valid_mask=image.patch_valid_mask,
                spatial_shapes=image.spatial_shapes,
                native_image_size=image.native_image_size,
                processed_patch_grid=image.processed_patch_grid,
                transform_hash=image.transform_hash,
            )
            embeddings = torch.nn.functional.normalize(temporal.pair_cls.float(), dim=-1).cpu()
        elapsed = time.perf_counter() - started
        probe = _probe_accuracy(embeddings, labels_tensor, args.seed)
        result["budgets"][str(budget)] = {
            "native_patch_tokens": list(image.patch_tokens.shape),
            "temporal_patch_tokens": list(temporal.temporal_patch_tokens.shape),
            "embedding_shape": list(embeddings.shape),
            "effective_rank": _effective_rank(embeddings),
            "embedding_norm_mean": float(embeddings.norm(dim=-1).mean()),
            "source_probe": probe,
            "seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del image, temporal, embeddings
        torch.cuda.empty_cache()
    result["source_shortcut_detected"] = any(
        float(value["source_probe"]["mean_accuracy"])
        >= float(result["source_shortcut_threshold"])
        for value in result["budgets"].values()
    )
    if result["source_shortcut_detected"]:
        result["status"] = "SOURCE_SHORTCUT_DETECTED"
    (output / "source_shortcut_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_shortcut_audit.sha256").write_text(
        f"{_sha256(output / 'source_shortcut_audit.json')}  source_shortcut_audit.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI failure contract
        print(f"SOURCE_SHORTCUT_AUDIT_FAILED: {exc}", file=sys.stderr)
        raise
