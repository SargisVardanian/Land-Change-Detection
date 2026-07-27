from __future__ import annotations

from pathlib import Path
import argparse

from ucv2_stage1_next_memory_core import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("universat_source", type=Path)
    parser.add_argument("universat_checkpoint", type=Path)
    parser.add_argument("jina_model", type=Path)
    parser.add_argument("temporal_depth", type=int, nargs="?", default=6)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--enable-patch-reranker", action="store_true")
    parser.add_argument("--qcpr-architecture-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--enable-temporal-explanation-channels", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.data_root,
            args.output_dir,
            args.universat_source,
            args.universat_checkpoint,
            args.jina_model,
            temporal_depth=args.temporal_depth,
            text_max_length=args.text_max_length,
            enable_patch_reranker=args.enable_patch_reranker,
            qcpr_architecture_version=args.qcpr_architecture_version,
            enable_temporal_explanation_channels=args.enable_temporal_explanation_channels,
        )
    )
