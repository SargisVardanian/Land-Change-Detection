from __future__ import annotations

import argparse
from pathlib import Path

from ucv2_stage1_next_core import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("universat_source", type=Path)
    parser.add_argument("universat_checkpoint", type=Path)
    parser.add_argument("jina_model", type=Path)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("num_workers", type=int)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initialize-from-v1", type=Path, default=None)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--max-captions-per-pair", type=int, default=2)
    parser.add_argument("--caption-frequency-power", type=float, default=0.5)
    parser.add_argument("--train-eval-pairs", type=int, default=1024)
    parser.add_argument("--train-eval-interval", type=int, default=2)
    parser.add_argument("--enable-conflict-filtering", action="store_true")
    parser.add_argument("--train-manifest", type=Path, action="append", default=[])
    parser.add_argument("--val-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--dataset-weight", action="append", default=None)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-improvement", type=float, default=0.002)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--enable-patch-reranker", action="store_true")
    parser.add_argument("--qcpr-architecture-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--enable-temporal-explanation-channels", action="store_true")
    parser.add_argument("--qcpr-alpha", type=float, default=1.0)
    parser.add_argument("--qcpr-beta", type=float, default=0.25)
    parser.add_argument("--qcpr-local-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-segmentation-loss-weight", type=float, default=0.2)
    parser.add_argument("--structured-fna-weight", type=float, default=0.0)
    parser.add_argument("--local-margin-loss-weight", type=float, default=0.1)
    parser.add_argument("--local-margin", type=float, default=0.1)
    parser.add_argument("--conditional-instance-loss-weight", type=float, default=0.01)
    parser.add_argument("--structured-auxiliary-loss-weight", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run(
            args.data_root,
            args.output_dir,
            args.universat_source,
            args.universat_checkpoint,
            args.jina_model,
            args.batch_size,
            args.epochs,
            args.num_workers,
            args.resume,
            initialize_from_v1=args.initialize_from_v1,
            temporal_depth=args.temporal_depth,
            max_captions_per_pair=args.max_captions_per_pair,
            caption_frequency_power=args.caption_frequency_power,
            train_eval_pairs=args.train_eval_pairs,
            train_eval_interval=args.train_eval_interval,
            enable_conflict_filtering=args.enable_conflict_filtering,
            train_manifests=tuple(args.train_manifest),
            val_manifests=tuple(args.val_manifest),
            dataset_config=args.dataset_config,
            dataset_sampling_weights=tuple(args.dataset_weight or ("levir_mci=0.55", "second_cc=0.45")),
            text_max_length=args.text_max_length,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_improvement=args.early_stopping_min_improvement,
            max_steps=args.max_steps,
            enable_patch_reranker=args.enable_patch_reranker,
            qcpr_architecture_version=args.qcpr_architecture_version,
            enable_temporal_explanation_channels=args.enable_temporal_explanation_channels,
            qcpr_alpha=args.qcpr_alpha,
            qcpr_beta=args.qcpr_beta,
            qcpr_local_loss_weight=args.qcpr_local_loss_weight,
            query_segmentation_loss_weight=args.query_segmentation_loss_weight,
            structured_fna_weight=args.structured_fna_weight,
            local_margin_loss_weight=args.local_margin_loss_weight,
            local_margin=args.local_margin,
            conditional_instance_loss_weight=args.conditional_instance_loss_weight,
            structured_auxiliary_loss_weight=args.structured_auxiliary_loss_weight,
        )
    )
