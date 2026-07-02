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
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--max-captions-per-pair", type=int, default=2)
    parser.add_argument("--caption-frequency-power", type=float, default=0.5)
    parser.add_argument("--train-eval-pairs", type=int, default=1024)
    parser.add_argument("--train-eval-interval", type=int, default=2)
    parser.add_argument("--enable-conflict-filtering", action="store_true")
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
            temporal_depth=args.temporal_depth,
            max_captions_per_pair=args.max_captions_per_pair,
            caption_frequency_power=args.caption_frequency_power,
            train_eval_pairs=args.train_eval_pairs,
            train_eval_interval=args.train_eval_interval,
            enable_conflict_filtering=args.enable_conflict_filtering,
        )
    )
