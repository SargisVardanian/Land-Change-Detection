from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from land_change_detection.backbones.jina_v5_text import JinaV5TextConfig, JinaV5TextEncoder
from land_change_detection.run_metadata import path_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline probe for Jina v5 retrieval text encoder.")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models" / "jina-v5-text-small-retrieval")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "model_probes" / "jina_v5.json")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        "new buildings appeared near an existing road",
        "новые здания появились около дороги",
        "ճանապարհի մոտ հայտնվել են նոր շենքեր",
    ]
    report: dict = {
        "repo_id": "jinaai/jina-embeddings-v5-text-small-retrieval",
        "local_path": str(args.model_path),
        "max_length": 64,
        "roles": ["query", "document"],
        "status": "not_started",
    }
    try:
        encoder = JinaV5TextEncoder(JinaV5TextConfig(model_path=args.model_path)).to(args.device)
        with torch.no_grad():
            query = encoder(samples, role="query")
            document = encoder(samples, role="document")
        report.update(
            {
                "status": "ok",
                "parameter_count": encoder.model_parameter_count,
                "dtype": encoder.dtype,
                "fingerprint": path_fingerprint(args.model_path),
                "query_global_shape": list(query.global_embedding.shape),
                "query_token_shape": list(query.token_embeddings.shape),
                "document_global_shape": list(document.global_embedding.shape),
                "document_token_shape": list(document.token_embeddings.shape),
                "query_norm_min": float(query.global_embedding.norm(dim=-1).min().item()),
                "query_norm_max": float(query.global_embedding.norm(dim=-1).max().item()),
                "sample_languages": ["en", "ru", "hy"],
                "metadata": query.metadata,
            }
        )
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
