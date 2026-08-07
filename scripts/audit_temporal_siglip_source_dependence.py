#!/usr/bin/env python3
"""Audit source shortcuts in frozen TemporalSigLIP pair embeddings.

This is an evaluation-only diagnostic.  It never updates the retrieval model;
the small ridge probe is fit on CPU to estimate how predictable LEVIR/SECOND
source identity is from the frozen gallery embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from qcpr_siglip2.data.manifest import group_rows_by_pair, load_exact_core_rows, load_exact_pair_rows
from qcpr_siglip2.data.runtime import encode_real_images
from qcpr_temporal_siglip.backbone import TemporalSigLIPBackbone
from qcpr_temporal_siglip.config import TemporalSigLIPConfig
from qcpr_temporal_siglip.model import TemporalSigLIP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--ranking",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--pair-batch-size", type=int, default=32)
    return parser.parse_args()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid named path: {value}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"invalid named path: {value}")
        result[name] = Path(raw_path)
    return result


def verify_code(expected: str) -> dict[str, Any]:
    worktree = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        ).strip()
    )
    if head != expected or dirty:
        raise RuntimeError("RUNTIME_CODE_STATE_MISMATCH")
    return {"head": head, "worktree_clean": True}


def ridge_probe_accuracy(features: torch.Tensor, labels: torch.Tensor, folds: int = 5) -> dict[str, Any]:
    """Deterministic stratified cross-validated binary ridge linear probe."""

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have incompatible shapes")
    classes = sorted(set(int(value) for value in labels.tolist()))
    if classes != [0, 1]:
        raise ValueError("source probe requires exactly two classes")
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    fold_ids = torch.full((features.shape[0],), -1, dtype=torch.long)
    for class_id in classes:
        indices = torch.where(labels == class_id)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        for offset, index in enumerate(indices.tolist()):
            fold_ids[index] = offset % folds
    accuracies: list[float] = []
    confusion = torch.zeros((2, 2), dtype=torch.long)
    for fold in range(folds):
        train = fold_ids != fold
        test = fold_ids == fold
        if not bool(train.any()) or not bool(test.any()):
            raise ValueError("empty source-probe fold")
        mean_value = features[train].mean(dim=0, keepdim=True)
        scale = features[train].std(dim=0, keepdim=True).clamp_min(1e-5)
        x_train = (features[train] - mean_value) / scale
        x_test = (features[test] - mean_value) / scale
        y_train = labels[train].float().unsqueeze(1)
        ones_train = torch.ones((x_train.shape[0], 1), dtype=x_train.dtype)
        x_aug = torch.cat((x_train, ones_train), dim=1)
        ridge = 1e-2 * torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
        ridge[-1, -1] = 0.0
        weights = torch.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y_train)
        ones_test = torch.ones((x_test.shape[0], 1), dtype=x_test.dtype)
        scores = torch.cat((x_test, ones_test), dim=1) @ weights
        predictions = (scores[:, 0] >= 0.5).long()
        targets = labels[test]
        accuracies.append(float((predictions == targets).float().mean()))
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            confusion[target, prediction] += 1
    return {
        "probe": "5_fold_stratified_ridge_linear_probe",
        "folds": folds,
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "accuracy_std": float(torch.tensor(accuracies).std(unbiased=False)),
        "fold_accuracies": accuracies,
        "confusion_matrix_true_rows": confusion.tolist(),
    }


def centroid_diagnostics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    class_features = [normalized[labels == class_id] for class_id in (0, 1)]
    centroids = [torch.nn.functional.normalize(value.mean(dim=0), dim=0) for value in class_features]
    similarity = normalized @ normalized.T
    same_mask = labels[:, None].eq(labels[None, :])
    off_diagonal = ~torch.eye(labels.numel(), dtype=torch.bool)
    same_values = similarity[same_mask & off_diagonal]
    cross_values = similarity[~same_mask]
    return {
        "centroid_cosine": float(centroids[0] @ centroids[1]),
        "centroid_l2_distance": float((centroids[0] - centroids[1]).norm()),
        "within_source_cosine_mean": float(same_values.mean()),
        "cross_source_cosine_mean": float(cross_values.mean()),
        "pair_norm_mean": float(features.float().norm(dim=-1).mean()),
        "pair_norm_std": float(features.float().norm(dim=-1).std(unbiased=False)),
        "effective_rank": int(torch.linalg.matrix_rank(features.float()).item()),
    }


def rank_metrics_for_source(
    rankings: torch.Tensor,
    query_rows: list[dict[str, Any]],
    pair_ids: list[str],
) -> dict[str, Any]:
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    source_ranks: dict[str, list[int]] = {}
    for query_index, row in enumerate(query_rows):
        positive_indices = [
            pair_index[str(value)]
            for value in row.get("positive_pair_ids", [])
            if str(value) in pair_index
        ]
        if not positive_indices:
            raise ValueError(f"query has no positive in gallery: {row['caption_id']}")
        ranked = rankings[query_index].tolist()
        rank = min(ranked.index(index) + 1 for index in positive_indices)
        source_ranks.setdefault(str(row["dataset_name"]), []).append(rank)
    result: dict[str, Any] = {}
    for source, ranks in sorted(source_ranks.items()):
        tensor = torch.tensor(ranks, dtype=torch.float32)
        result[source] = {
            "query_count": len(ranks),
            "mrr_full": float((1.0 / tensor).mean()),
            "hit_at_10": float((tensor <= 10).float().mean()),
            "median_rank": float(tensor.median()),
        }
    return result


def main() -> int:
    args = parse_args()
    code = verify_code(args.expected_code_sha)
    checkpoints = parse_named_paths(args.checkpoint)
    rankings = parse_named_paths(args.ranking)
    if set(checkpoints) != set(rankings):
        raise ValueError("checkpoint and ranking names must match")
    manifest = Path(args.development_manifest)
    core_rows = load_exact_core_rows(manifest, split="development")
    query_rows = load_exact_pair_rows(manifest, split="development")
    groups = group_rows_by_pair(core_rows)
    pair_rows = [group[0] for group in groups.values()]
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    sources = sorted({str(row["dataset_name"]) for row in pair_rows})
    source_labels = torch.tensor(
        [sources.index(str(row["dataset_name"])) for row in pair_rows], dtype=torch.long
    )
    if sources != ["levir_mci", "second_cc"]:
        raise ValueError(f"unexpected source labels: {sources}")
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.siglip2_model, local_files_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "PASS",
        "code": code,
        "data_release": str(args.data_release),
        "development_manifest": str(manifest),
        "development_manifest_sha256": sha256(manifest),
        "gallery_size": len(pair_rows),
        "exact_query_count": len(query_rows),
        "source_counts": {source: int((source_labels == index).sum()) for index, source in enumerate(sources)},
        "arms": {},
        "interpretation_policy": {
            "source_probe_is_diagnostic": True,
            "high_source_predictability_is_not_alone_a_failure": True,
            "flag_requires_comparison_with_retrieval_gain": True,
        },
    }
    for name, checkpoint_path in sorted(checkpoints.items()):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        backbone = TemporalSigLIPBackbone(
            args.siglip2_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model = TemporalSigLIP(backbone, TemporalSigLIPConfig()).to(device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise ValueError(f"invalid checkpoint: {checkpoint_path}")
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(pair_rows), args.pair_batch_size):
                image = encode_real_images(
                    backbone,
                    processor,
                    pair_rows[start : start + args.pair_batch_size],
                    device,
                    dtype=torch.bfloat16,
                    no_grad=True,
                )
                embeddings.append(
                    model.encode_pair_from_features(image.patch_tokens).pair_embedding.float().cpu()
                )
        pair_embeddings = torch.cat(embeddings, dim=0)
        embedding_path = output_dir / f"{name}_pair_embeddings.pt"
        torch.save(pair_embeddings, embedding_path)
        ranking = torch.load(rankings[name], map_location="cpu", weights_only=True)
        if not isinstance(ranking, torch.Tensor) or tuple(ranking.shape) != (len(query_rows), len(pair_rows)):
            raise ValueError(f"ranking shape mismatch: {rankings[name]}")
        probe = ridge_probe_accuracy(pair_embeddings, source_labels)
        centroids = centroid_diagnostics(pair_embeddings, source_labels)
        report["arms"][name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "ranking": str(rankings[name]),
            "ranking_sha256": sha256(rankings[name]),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256(embedding_path),
            "source_linear_probe": probe,
            "centroid_and_embedding": centroids,
            "source_conditioned_rank_metrics": rank_metrics_for_source(ranking, query_rows, pair_ids),
        }
        del model, backbone, pair_embeddings
        torch.cuda.empty_cache()
    stage_a = report["arms"].get("stage_a")
    stage_b = report["arms"].get("stage_b_step2048")
    if stage_a is not None and stage_b is not None:
        accuracy_delta = (
            stage_b["source_linear_probe"]["accuracy_mean"]
            - stage_a["source_linear_probe"]["accuracy_mean"]
        )
        report["comparison"] = {
            "stage_b_minus_stage_a_probe_accuracy": accuracy_delta,
            "source_dependence_flag": "REVIEW_REQUIRED" if accuracy_delta > 0.10 else "NO_LARGE_DELTA_DETECTED",
        }
    output = output_dir / "source_separability.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
