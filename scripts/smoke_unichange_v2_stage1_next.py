from __future__ import annotations

import argparse
from pathlib import Path

from ucv2_cluster_report import finalize
from ucv2_stage1_next_smoke_core import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("universat_source", type=Path)
    parser.add_argument("universat_checkpoint", type=Path)
    parser.add_argument("jina_model", type=Path)
    parser.add_argument("temporal_depth", type=int, nargs="?", default=6)
    parser.add_argument("--train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--val-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--dataset-weight", action="append", default=None)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--enable-patch-reranker", action="store_true")
    parser.add_argument("--qcpr-architecture-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--enable-temporal-explanation-channels", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    temporal_depth = args.temporal_depth
    result = run(
        args.data_root,
        output_dir,
        args.universat_source,
        args.universat_checkpoint,
        args.jina_model,
        temporal_depth=temporal_depth,
        train_manifests=tuple(args.train_manifest),
        val_manifests=tuple(args.val_manifest),
        dataset_config=args.dataset_config,
        dataset_sampling_weights=tuple(args.dataset_weight or ("levir_mci=0.55", "second_cc=0.45")),
        text_max_length=args.text_max_length,
        enable_patch_reranker=args.enable_patch_reranker,
        qcpr_architecture_version=args.qcpr_architecture_version,
        enable_temporal_explanation_channels=args.enable_temporal_explanation_channels,
    )
    report = finalize(output_dir, "cuda")
    required = (
        report.get("real_cluster_smoke_passed")
        and report.get("stage1_next") is True
        and report.get("loss") == "semantic_soft_target_text_to_pair"
        and report.get("stable_caption_groups") is True
        and report.get("use_direction_embeddings") is True
        and report.get("use_explicit_change_fusion") is True
        and report.get("trainable_temperature") is True
        and report.get("temporal_depth") == temporal_depth
        and report.get("text_adapter_enabled") is True
        and report.get("text_max_length") == args.text_max_length
        and report.get("patch_reranker_available") is args.enable_patch_reranker
        and (not args.enable_patch_reranker or report.get("qcpr_gradient_audit_passed") is True)
    )
    if not required:
        raise SystemExit("Stage-1-next smoke report did not pass all feature gates")
    raise SystemExit(result)
