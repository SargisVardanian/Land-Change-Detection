from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from land_change_detection.retrieval.backends.pair_analog_prithvi import build_pair_manifest_item
from land_change_detection.retrieval.backends.static_region_prithvi import (
    build_static_manifest_item,
    create_embedding_backend,
    vector_store_name,
)


def _read_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "items" in payload:
            return [dict(item) for item in payload["items"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError("JSON manifest must contain a list or an 'items' list.")
    payloads: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payloads.append(dict(json.loads(line)))
    return payloads


def build_index(mode: str, input_manifest: Path, output_path: Path, embedding_backend_name: str, prefer_faiss: bool) -> dict[str, Any]:
    embedding_backend = create_embedding_backend(embedding_backend_name)
    payloads = _read_payloads(input_manifest)
    items = []
    for payload in payloads:
        if mode == "static_region":
            items.append(build_static_manifest_item(payload, embedding_backend).to_dict())
        elif mode == "pair_analog":
            items.append(build_pair_manifest_item(payload, embedding_backend).to_dict())
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
    output = {
        "mode": mode,
        "embedding_backend": embedding_backend.name,
        "vector_store": vector_store_name(prefer_faiss),
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic retrieval index from a small manifest.")
    parser.add_argument("--mode", choices=("static_region", "pair_analog"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-backend", choices=("fake", "numpy"), default="numpy")
    parser.add_argument("--prefer-faiss", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_index(
        mode=args.mode,
        input_manifest=args.input_manifest,
        output_path=args.output,
        embedding_backend_name=args.embedding_backend,
        prefer_faiss=args.prefer_faiss,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
