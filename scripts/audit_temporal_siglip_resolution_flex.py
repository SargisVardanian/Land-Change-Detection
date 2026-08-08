#!/usr/bin/env python3
"""Run the CPU resolution-flexibility contract without touching training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel


def run_contract(output: Path) -> dict[str, object]:
    config = Siglip2TemporalConfig(
        hidden_size=32,
        attention_heads=4,
        mlp_size=128,
        evidence_query_chunk_size=2,
        evidence_pair_chunk_size=2,
        large_scene_latents=8,
        direct_patch_token_budget=1024,
    ).validate()
    model = Siglip2TemporalRetrievalModel(None, config).eval()
    torch.manual_seed(20260808)
    results: dict[str, object] = {"config": config.to_dict(), "budgets": {}}
    for budget in (256, 576, 1024):
        visual = torch.randn(1, 2, budget, 32)
        pooled = torch.randn(1, 2, 32)
        text_tokens = torch.randn(2, 5, 32)
        text_embedding = torch.randn(2, 32)
        text_mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            value = model.forward_from_features(
                visual, pooled, text_tokens, text_embedding, text_mask
            )
        results["budgets"][str(budget)] = {
            "pair_embedding_shape": list(value.pair_cls.shape),
            "dense_token_shape": list(value.temporal.temporal_patch_tokens.shape),
            "evidence_map_shape": list(value.evidence.evidence_map.shape),
            "finite": bool(torch.isfinite(value.score_matrix).all()),
            "single_vector_per_item": value.pair_cls.shape == (1, 32),
        }
    reducer_model = Siglip2TemporalRetrievalModel(
        None,
        Siglip2TemporalConfig(
            hidden_size=32,
            attention_heads=4,
            mlp_size=128,
            evidence_query_chunk_size=2,
            evidence_pair_chunk_size=2,
            large_scene_latents=8,
            direct_patch_token_budget=64,
        ).validate(),
    ).eval()
    large_visual = torch.randn(1, 2, 128, 32)
    with torch.no_grad():
        large = reducer_model.forward_from_features(
            large_visual,
            torch.randn(1, 2, 32),
            torch.randn(2, 5, 32),
            torch.randn(2, 32),
            torch.ones(2, 5, dtype=torch.bool),
        )
    results["large_scene"] = {
        "reduced": large.temporal.reduced,
        "processed_patch_count": large.temporal.patch_count,
        "pair_embedding_shape": list(large.pair_cls.shape),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_contract(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
