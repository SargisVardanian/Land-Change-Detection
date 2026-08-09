#!/usr/bin/env python3
"""Compare fixed-resolution and NaFlex SigLIP-2 runtime on one exact gallery.

This is a compatibility evaluation only.  It restores the same temporal and
evidence adapter checkpoint on two independently loaded frozen backbones and
uses identical physical pairs and exact queries.  It does not train either
backbone or claim a scientific retrieval improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
from qcpr_siglip2.data.manifest import load_exact_core_rows, ordered_id_sha256
from qcpr_siglip2.data.runtime import (
    encode_real_images,
    encode_real_text,
    processor_text_inputs,
)
from qcpr_siglip2.evaluation.bootstrap import paired_clustered_rank_bootstrap
from qcpr_siglip2.evaluation.common_gallery import (
    canonical_pair_rows,
    exact_relevance_masks,
    metrics_by_query_group,
)
from qcpr_siglip2.evaluation.retrieval import first_positive_ranks, full_gallery_metrics
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _restore_adapter(
    model: Siglip2TemporalRetrievalModel, checkpoint: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain model_state")
    adapter_state = {
        key: value for key, value in state.items() if not key.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    non_backbone_missing = [key for key in missing if not key.startswith("backbone.")]
    if unexpected or non_backbone_missing:
        raise RuntimeError(
            "adapter checkpoint mismatch: "
            f"missing={non_backbone_missing[:8]} unexpected={unexpected[:8]}"
        )
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "restored_adapter_keys": len(adapter_state),
    }


def _encode_single_vector_gallery(
    model: Siglip2TemporalRetrievalModel,
    backbone: Siglip2Backbone,
    processor: Any,
    pair_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    image_batch_size: int,
    text_batch_size: int,
    max_num_patches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pair_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    visual_shapes: list[list[int]] = []
    for start in range(0, len(pair_rows), image_batch_size):
        image = encode_real_images(
            backbone,
            processor,
            pair_rows[start : start + image_batch_size],
            device,
            max_num_patches=max_num_patches,
            is_naflex=backbone.is_naflex,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            temporal = model.temporal_adapter(
                image.patch_tokens,
                image.pooled_embedding,
                patch_valid_mask=image.patch_valid_mask,
                spatial_shapes=image.spatial_shapes,
                native_image_size=image.native_image_size,
                processed_patch_grid=image.processed_patch_grid,
                transform_hash=image.transform_hash,
                token_coordinates=image.token_coordinates,
            )
        pair_embeddings.append(temporal.pair_cls.float().cpu())
        visual_shapes.append(list(image.patch_tokens.shape))
    for start in range(0, len(query_rows), text_batch_size):
        text = encode_real_text(
            backbone,
            processor,
            query_rows[start : start + text_batch_size],
            device,
        )
        text_embeddings.append(
            torch.nn.functional.normalize(text.pooled_embedding.float(), dim=-1).cpu()
        )
    return (
        torch.cat(pair_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        {"visual_batch_shapes": visual_shapes},
    )


def _constructor_defaults(config: Any) -> dict[str, Any]:
    signature = inspect.signature(type(config))
    return {
        name: signature.parameters[name].default
        for name in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")
        if name in signature.parameters
    }


def _tokenizer_config_gate(
    backbone: Siglip2Backbone,
    processor: Any,
    model_path: Path,
    *,
    repository: str,
    revision: str,
    query_rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Audit raw, constructor-default and actual runtime token contracts."""

    source = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_inputs = processor_text_inputs(processor, query_rows, device)
    input_ids = text_inputs["input_ids"].detach().cpu()
    attention = text_inputs["attention_mask"].detach().cpu().bool()
    valid_ids = input_ids[attention]
    runtime = dict(backbone.tokenizer_contract)
    embedding_vocab = int(runtime["model_embedding_vocab_size"])
    processor_tokenizer = getattr(processor, "tokenizer", None)
    processor_ids = {
        name: getattr(processor_tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    constructor_defaults = _constructor_defaults(backbone.text_model.config)
    warning_from_defaults = bool(
        isinstance(constructor_defaults.get("vocab_size"), int)
        and any(
            isinstance(constructor_defaults.get(name), int)
            and constructor_defaults[name] >= constructor_defaults["vocab_size"]
            for name in ("bos_token_id", "eos_token_id")
        )
    )
    passed = bool(
        runtime.get("runtime_config_ids_valid")
        and valid_ids.numel() > 0
        and int(valid_ids.min()) >= 0
        and int(valid_ids.max()) < embedding_vocab
        and processor_ids == runtime.get("special_token_ids")
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "repository": repository,
        "revision": revision,
        "local_path": str(model_path),
        "config_sha256": _sha256(model_path / "config.json"),
        "tokenizer_config_sha256": _sha256(model_path / "tokenizer_config.json"),
        "weights_sha256": _sha256(model_path / "model.safetensors"),
        "config_class": type(backbone.model.config).__name__,
        "text_config_class": type(backbone.text_model.config).__name__,
        "runtime_model_class": backbone.runtime_class,
        "raw_text_config": source.get("text_config", {}),
        "upstream_constructor_defaults": constructor_defaults,
        "warning_provenance": {
            "emitter": f"{type(backbone.text_model.config).__name__} constructor defaults",
            "out_of_range_defaults": warning_from_defaults,
            "scientific_text_path_uses_defaults": False,
            "runtime_contract_validated_after_load": passed,
            "warning_suppressed": False,
        },
        "tokenizer": {
            "class": type(backbone.tokenizer).__name__,
            "processor_class": type(processor).__name__,
            "processor_tokenizer_class": type(processor_tokenizer).__name__,
            "vocab_size": int(backbone.tokenizer.vocab_size),
            "special_token_ids": runtime.get("special_token_ids"),
            "processor_special_token_ids": processor_ids,
            "model_max_length": getattr(backbone.tokenizer, "model_max_length", None),
        },
        "embedding_vocab_size": embedding_vocab,
        "actual_batch": {
            "query_count": len(query_rows),
            "input_shape": list(input_ids.shape),
            "minimum_valid_input_id": int(valid_ids.min()),
            "maximum_valid_input_id": int(valid_ids.max()),
            "attention_mask_shape": list(attention.shape),
            "attention_token_count": int(attention.sum()),
            "max_sequence_length": int(input_ids.shape[1]),
            "all_valid_ids_in_embedding_range": bool(
                int(valid_ids.min()) >= 0 and int(valid_ids.max()) < embedding_vocab
            ),
        },
        "runtime_contract": runtime,
    }


def _linear_source_probe(
    embeddings: torch.Tensor, labels: torch.Tensor, *, seed: int
) -> tuple[float, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(labels.numel(), generator=generator)
    scores: list[float] = []
    predictions = torch.empty_like(labels)
    for fold in range(5):
        test = order[fold::5]
        train = order[~torch.isin(torch.arange(labels.numel()), test)]
        x_train = embeddings[train].float()
        x_test = embeddings[test].float()
        mean = x_train.mean(0, keepdim=True)
        scale = x_train.std(0, keepdim=True).clamp_min(1e-6)
        x_train = torch.cat([(x_train - mean) / scale, torch.ones(len(train), 1)], 1)
        x_test = torch.cat([(x_test - mean) / scale, torch.ones(len(test), 1)], 1)
        regularizer = torch.eye(x_train.shape[1]) * 1e-2
        regularizer[-1, -1] = 0.0
        weights = torch.linalg.solve(
            x_train.T @ x_train + regularizer,
            x_train.T @ labels[train].float(),
        )
        fold_predictions = (x_test @ weights >= 0.5).long()
        predictions[test] = fold_predictions
        scores.append(float((fold_predictions == labels[test]).float().mean()))
    return float(torch.tensor(scores).mean()), predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naflex-model", required=True)
    parser.add_argument("--fixres-model", required=True)
    parser.add_argument("--naflex-repository", required=True)
    parser.add_argument("--fixres-repository", required=True)
    parser.add_argument("--naflex-revision", required=True)
    parser.add_argument("--fixres-revision", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
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
    core_rows = load_exact_core_rows(manifest, split="development")
    query_rows = list(core_rows)
    pair_rows = canonical_pair_rows(core_rows)
    if args.max_pairs > 0:
        selected_ids = {
            str(row["canonical_pair_id"]) for row in pair_rows[: args.max_pairs]
        }
        pair_rows = pair_rows[: args.max_pairs]
        query_rows = [
            row for row in query_rows if str(row["canonical_pair_id"]) in selected_ids
        ]
    if len(pair_rows) < 4 or not query_rows:
        raise ValueError("compatibility comparison needs at least four pairs and queries")
    pair_ids = [str(row["canonical_pair_id"]) for row in pair_rows]
    positive, ignored = exact_relevance_masks(query_rows, pair_rows)
    exact_selector = torch.tensor(
        [row.get("query_scope") == "exact_pair" for row in query_rows],
        dtype=torch.bool,
    )
    if not exact_selector.any():
        raise ValueError("compatibility comparison has no primary exact queries")
    config = _load_config(Path(args.config_path))
    result: dict[str, Any] = {
        "status": "FULL_GALLERY_REGRESSION_PENDING",
        "evaluation_only": True,
        "code": state,
        "expected_code_sha": args.expected_code_sha,
        "release": str(release),
        "development_manifest": str(manifest),
        "development_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "pair_count": len(pair_rows),
        "query_count": len(query_rows),
        "exact_primary_query_count": int(exact_selector.sum()),
        "non_exact_diagnostic_query_count": int((~exact_selector).sum()),
        "pair_ids": pair_ids,
        "ordered_pair_ids_sha256": ordered_id_sha256(
            pair_rows, "canonical_pair_id"
        ),
        "ordered_query_ids_sha256": ordered_id_sha256(query_rows, "caption_id"),
        "config": config.to_dict(),
        "acceptance_contract": {
            "mrr_ratio_minimum": 0.85,
            "candidate_hit_at_10_max_absolute_regression": 0.02,
            "median_rank_ratio_maximum": 1.15,
            "defined_before_full_gallery_result": True,
        },
        "models": {},
    }
    device = torch.device("cuda")
    outputs: dict[str, torch.Tensor] = {}
    embeddings_by_model: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, model_path, repository, revision, budget in (
        (
            "naflex_256",
            args.naflex_model,
            args.naflex_repository,
            args.naflex_revision,
            256,
        ),
        (
            "fixres_256",
            args.fixres_model,
            args.fixres_repository,
            args.fixres_revision,
            None,
        ),
    ):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        backbone = Siglip2Backbone(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Siglip2TemporalRetrievalModel(backbone, config).to(device).eval()
        restore = _restore_adapter(model, checkpoint)
        tokenizer_gate = _tokenizer_config_gate(
            backbone,
            processor,
            Path(model_path),
            repository=repository,
            revision=revision,
            query_rows=query_rows[: args.text_batch_size],
            device=device,
        )
        if tokenizer_gate["status"] != "PASS":
            raise RuntimeError(f"TOKENIZER_CONFIG_GATE_FAIL:{name}")
        pair_embeddings, text_embeddings, encoding_contract = (
            _encode_single_vector_gallery(
                model,
                backbone,
                processor,
                pair_rows,
                query_rows,
                device=device,
                image_batch_size=args.image_batch_size,
                text_batch_size=args.text_batch_size,
                max_num_patches=budget,
            )
        )
        scores = (
            text_embeddings @ pair_embeddings.transpose(0, 1)
            / float(model.retrieval_temperature.detach().cpu())
        ).masked_fill(ignored, -1.0e4)
        torch.cuda.synchronize(device)
        outputs[name] = scores
        embeddings_by_model[name] = (pair_embeddings, text_embeddings)
        result["models"][name] = {
            "path": str(model_path),
            "runtime_class": backbone.runtime_class,
            "is_naflex": backbone.is_naflex,
            "encoding_contract": encoding_contract,
            "pair_embedding_shape": list(pair_embeddings.shape),
            "text_embedding_shape": list(text_embeddings.shape),
            "score_shape": list(scores.shape),
            "metrics": full_gallery_metrics(
                scores[exact_selector], positive[exact_selector]
            ),
            "all_query_diagnostic_metrics": full_gallery_metrics(scores, positive),
            "per_source_metrics": metrics_by_query_group(
                scores, positive, query_rows
            ),
            "adapter_restore": restore,
            "tokenizer_config_gate": tokenizer_gate,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del model, backbone, processor
        torch.cuda.empty_cache()
    tokenizer_report = {
        "status": (
            "PASS"
            if all(
                model["tokenizer_config_gate"]["status"] == "PASS"
                for model in result["models"].values()
            )
            else "FAIL"
        ),
        "models": {
            name: model["tokenizer_config_gate"]
            for name, model in result["models"].items()
        },
    }
    tokenizer_report_path = output / "reports" / "siglip2_tokenizer_config_gate.json"
    tokenizer_report_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_report_path.write_text(
        json.dumps(tokenizer_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["tokenizer_config_gate"] = {
        "status": tokenizer_report["status"],
        "path": str(tokenizer_report_path),
        "sha256": _sha256(tokenizer_report_path),
    }
    naflex_scores = outputs["naflex_256"]
    fixres_scores = outputs["fixres_256"]
    naflex_ranks = first_positive_ranks(naflex_scores, positive)
    fixres_ranks = first_positive_ranks(fixres_scores, positive)
    rankings_path = output / "full_rankings.pt"
    torch.save(
        {
            "naflex_256_scores": naflex_scores,
            "fixres_256_scores": fixres_scores,
            "naflex_256_ranks": naflex_ranks,
            "fixres_256_ranks": fixres_ranks,
            "positive_mask": positive,
            "ignored_mask": ignored,
            "pair_ids": pair_ids,
            "query_ids": [str(row["caption_id"]) for row in query_rows],
        },
        rankings_path,
    )
    result["full_rankings_sha256"] = _sha256(rankings_path)
    rank_records = []
    for index, row in enumerate(query_rows):
        record: dict[str, Any] = {
            "query_id": str(row["caption_id"]),
            "physical_pair_cluster_id": str(row["canonical_pair_id"]),
            "source": str(row["dataset_name"]),
            "exact_primary": bool(exact_selector[index]),
        }
        for model_name, ranks in (
            ("naflex_256", naflex_ranks),
            ("fixres_256", fixres_ranks),
        ):
            rank = int(ranks[index])
            record[model_name] = {
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                **{
                    f"candidate_hit_at_{k}": rank <= k
                    for k in (1, 5, 10, 50, 100)
                },
            }
        rank_records.append(record)
    with (output / "per_query_rank_statistics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in rank_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    result["per_query_rank_statistics_sha256"] = _sha256(
        output / "per_query_rank_statistics.jsonl"
    )
    prebootstrap = output / "resolution_compatibility_prebootstrap.json"
    prebootstrap.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolution_compatibility_prebootstrap.sha256").write_text(
        f"{_sha256(prebootstrap)}  {prebootstrap.name}\n", encoding="utf-8"
    )
    result["score_matrix_max_abs_difference"] = float((naflex_scores - fixres_scores).abs().max())
    result["score_matrix_pearson"] = float(
        torch.corrcoef(torch.stack([naflex_scores.flatten(), fixres_scores.flatten()]))[0, 1]
    )
    result["top1_agreement"] = float(
        (naflex_scores.argmax(dim=1) == fixres_scores.argmax(dim=1)).float().mean()
    )
    naflex_metrics = result["models"]["naflex_256"]["metrics"]
    fixres_metrics = result["models"]["fixres_256"]["metrics"]
    mrr_ratio = float(naflex_metrics["mrr_full"] / fixres_metrics["mrr_full"])
    hit10_delta = float(
        naflex_metrics["candidate_hit_at_10"]
        - fixres_metrics["candidate_hit_at_10"]
    )
    median_ratio = float(
        naflex_metrics["median_rank"] / fixres_metrics["median_rank"]
    )
    accepted = bool(
        mrr_ratio >= 0.85 and hit10_delta >= -0.02 and median_ratio <= 1.15
    )
    result["acceptance_result"] = {
        "mrr_ratio": mrr_ratio,
        "candidate_hit_at_10_delta": hit10_delta,
        "median_rank_ratio": median_ratio,
        "passed": accepted,
    }
    exact_indices = exact_selector.nonzero(as_tuple=False).flatten()
    result["paired_physical_pair_bootstrap"] = paired_clustered_rank_bootstrap(
        naflex_ranks[exact_indices],
        fixres_ranks[exact_indices],
        [str(query_rows[index]["canonical_pair_id"]) for index in exact_indices.tolist()],
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    result["paired_physical_pair_bootstrap"][
        "delta_definition"
    ] = "naflex_256_minus_fixres_256"
    pair_sources = [str(row["dataset_name"]) for row in pair_rows]
    query_sources = [str(row["dataset_name"]) for row in query_rows]
    source_names = sorted(set(pair_sources))
    if len(source_names) != 2 or set(query_sources) != set(source_names):
        raise ValueError("source audit requires exactly LEVIR-MCI and SECOND-CC")
    pair_labels = torch.tensor([source_names.index(value) for value in pair_sources])
    query_labels = torch.tensor([source_names.index(value) for value in query_sources])
    source_match = query_labels[:, None] == pair_labels[None, :]
    source_audit: dict[str, Any] = {
        "sources": source_names,
        "probe_method": "five_fold_frozen_ridge_linear_probe",
        "models": {},
    }
    for name, scores in outputs.items():
        pair_embeddings, text_embeddings = embeddings_by_model[name]
        restricted = scores.masked_fill(~source_match, -1.0e4)
        wrong_source_decoys = scores.masked_fill(
            source_match & ~positive, -1.0e4
        )
        full = full_gallery_metrics(scores[exact_selector], positive[exact_selector])
        within = full_gallery_metrics(
            restricted[exact_selector], positive[exact_selector]
        )
        decoys = full_gallery_metrics(
            wrong_source_decoys[exact_selector], positive[exact_selector]
        )
        pair_probe_accuracy, _pair_predictions = _linear_source_probe(
            pair_embeddings, pair_labels, seed=args.bootstrap_seed
        )
        text_probe_accuracy, text_predictions = _linear_source_probe(
            text_embeddings, query_labels, seed=args.bootstrap_seed
        )
        pair_tie_break = torch.tensor(
            [
                int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:12], 16)
                / float(16**12)
                for pair_id in pair_ids
            ],
            dtype=torch.float64,
        )
        source_only_scores = (
            (text_predictions[:, None] == pair_labels[None, :]).double()
            + pair_tie_break[None, :] * 1e-6
        )
        source_only = full_gallery_metrics(
            source_only_scores[exact_selector], positive[exact_selector]
        )
        source_rank_distributions = {}
        model_ranks = first_positive_ranks(scores, positive)
        for source in source_names:
            indices = torch.tensor(
                [
                    index
                    for index, value in enumerate(query_sources)
                    if value == source and bool(exact_selector[index])
                ],
                dtype=torch.long,
            )
            values = model_ranks[indices].float()
            source_rank_distributions[source] = {
                "query_count": int(values.numel()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(torch.quantile(values, 0.10)),
                "p90": float(torch.quantile(values, 0.90)),
            }
        source_audit["models"][name] = {
            "pair_embedding_source_probe_accuracy": pair_probe_accuracy,
            "text_embedding_source_probe_accuracy": text_probe_accuracy,
            "full_gallery": full,
            "source_restricted_gallery": within,
            "true_pair_plus_wrong_source_decoys": decoys,
            "source_only_control": {
                "formula": "predicted_query_source_match + fixed_pair_hash_tiebreak",
                "uses_pair_semantics": False,
                "metrics": source_only,
            },
            "per_source_rank_distributions": source_rank_distributions,
            "source_restriction_mrr_gain": float(
                within["mrr_full"] - full["mrr_full"]
            ),
            "source_only_mrr_fraction_of_full": float(
                source_only["mrr_full"] / max(full["mrr_full"], 1e-12)
            ),
        }
    naflex_source = source_audit["models"]["naflex_256"]
    relative_gain = float(
        naflex_source["source_restriction_mrr_gain"]
        / max(naflex_source["full_gallery"]["mrr_full"], 1e-12)
    )
    source_only_fraction = float(naflex_source["source_only_mrr_fraction_of_full"])
    source_audit["source_separable"] = bool(
        naflex_source["pair_embedding_source_probe_accuracy"] >= 0.75
        or naflex_source["text_embedding_source_probe_accuracy"] >= 0.75
    )
    source_audit["source_restriction_relative_mrr_gain"] = relative_gain
    source_audit["source_only_mrr_fraction_of_full"] = source_only_fraction
    source_audit["retrieval_evidence"] = (
        "HIGH"
        if relative_gain >= 1.0 or source_only_fraction >= 0.75
        else "MEDIUM"
        if relative_gain >= 0.25 or source_only_fraction >= 0.25
        else "LOW"
    )
    source_audit["classification_rule"] = {
        "HIGH": "source restriction gain >= 100% or source-only MRR >= 75% of full",
        "MEDIUM": "source restriction gain >= 25% or source-only MRR >= 25% of full",
        "LOW": "both source-only criteria below MEDIUM thresholds",
    }
    result["source_shortcut_audit"] = source_audit
    result["status"] = (
        "NAFLEX_256_MATCHED_REGRESSION_PASS"
        if accepted
        else "NAFLEX_256_MATCHED_REGRESSION_FAIL"
    )
    (output / "resolution_compatibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolution_compatibility.sha256").write_text(
        f"{_sha256(output / 'resolution_compatibility.json')}  resolution_compatibility.json\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI failure contract
        print(f"RESOLUTION_COMPATIBILITY_FAILED: {exc}", file=sys.stderr)
        raise
