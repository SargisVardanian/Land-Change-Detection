from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.retrieval.backends.pair_analog_prithvi import PairAnalogPrithviBackend
from land_change_detection.retrieval.backends.static_region_prithvi import StaticRegionPrithviBackend
from land_change_detection.retrieval.contracts import RetrievalMode, RetrievalQuery


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a deterministic retrieval index.")
    parser.add_argument("--mode", choices=("static_region", "pair_analog"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-backend", choices=("fake", "numpy"), default="numpy")
    parser.add_argument("--prefer-faiss", action="store_true")
    parser.add_argument("--image-path")
    parser.add_argument("--before-image-path")
    parser.add_argument("--after-image-path")
    parser.add_argument("--item-id")
    parser.add_argument("--text")
    parser.add_argument("--month", type=int)
    parser.add_argument("--season")
    parser.add_argument("--sensor")
    parser.add_argument("--geography")
    parser.add_argument("--region-id")
    parser.add_argument("--transition-label")
    return parser


def _build_query(args: argparse.Namespace) -> RetrievalQuery:
    filters = {
        key: value
        for key, value in {
            "month": args.month,
            "season": args.season,
            "sensor": args.sensor,
            "geography": args.geography,
            "region_id": args.region_id,
            "transition_label": args.transition_label,
        }.items()
        if value is not None
    }
    return RetrievalQuery(
        mode=RetrievalMode(args.mode),
        top_k=args.top_k,
        text=args.text,
        item_id=args.item_id,
        image_path=args.image_path,
        before_image_path=args.before_image_path,
        after_image_path=args.after_image_path,
        filters=filters,
    )


def query_index(args: argparse.Namespace) -> dict:
    if args.mode == "static_region":
        backend = StaticRegionPrithviBackend(
            model_dir=args.index,
            device="cpu",
            embedding_backend_name=args.embedding_backend,
            prefer_faiss=args.prefer_faiss,
        )
    else:
        backend = PairAnalogPrithviBackend(
            model_dir=args.index,
            device="cpu",
            embedding_backend_name=args.embedding_backend,
            prefer_faiss=args.prefer_faiss,
        )
    artifact = backend.retrieve(_build_query(args))
    return artifact.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(query_index(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
