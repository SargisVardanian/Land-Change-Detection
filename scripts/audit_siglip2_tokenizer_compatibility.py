#!/usr/bin/env python3
"""Audit the pinned SigLIP-2 tokenizer/config/embedding contract.

This is intentionally an executable audit, not a warning-suppression wrapper.
It records the inherited checkpoint config, the tokenizer actually used by the
processor, and the post-construction runtime config after the model boundary
has validated and synchronized special-token ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qcpr_siglip2.backbones.siglip2 import Siglip2Backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--text",
        default="A new building appeared between the two dates.",
    )
    return parser.parse_args()


def load_source_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model config must be a JSON object")
    return value


def main() -> int:
    args = parse_args()
    model_path = Path(args.siglip2_model)
    source_config = load_source_config(model_path)
    source_text_config = source_config.get("text_config", {})
    if not isinstance(source_text_config, dict):
        raise ValueError("text_config must be an object")

    backbone = Siglip2Backbone(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    contract = dict(backbone.tokenizer_contract)
    tokenizer = backbone.tokenizer
    encoded = tokenizer(
        [args.text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    ids = encoded["input_ids"]
    attention = encoded.get("attention_mask")
    if attention is None:
        raise ValueError("tokenizer did not return attention_mask")
    valid_ids = ids[attention.bool()]
    embedding_vocab_size = int(contract["model_embedding_vocab_size"])
    token_ids_in_range = bool(
        valid_ids.numel() > 0
        and int(valid_ids.min()) >= 0
        and int(valid_ids.max()) < embedding_vocab_size
    )

    raw_ids = {
        name: source_text_config.get(name)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    tokenizer_ids = contract["special_token_ids"]
    inherited_config_warning = any(
        isinstance(value, int)
        and value >= 32000
        for name, value in raw_ids.items()
        if name in {"bos_token_id", "eos_token_id"}
    )
    report: dict[str, Any] = {
        "status": "PASS"
        if contract.get("runtime_config_ids_valid")
        and token_ids_in_range
        else "FAIL",
        "model_path": str(model_path),
        "runtime_class": backbone.runtime_class,
        "source_config": {
            "model_type": source_config.get("model_type"),
            "text_vocab_size": source_text_config.get("vocab_size"),
            "inherited_special_token_ids": raw_ids,
            "upstream_constructor_warning_expected": inherited_config_warning,
        },
        "tokenizer": {
            "class": contract.get("tokenizer_class"),
            "vocab_size": contract.get("tokenizer_vocab_size"),
            "special_token_ids": tokenizer_ids,
            "model_max_length": getattr(tokenizer, "model_max_length", None),
        },
        "embedding": {
            "vocab_size": embedding_vocab_size,
            "hidden_size": backbone.hidden_size,
        },
        "runtime": contract,
        "sample_tokenization": {
            "sequence_length": int(ids.shape[-1]),
            "min_valid_token_id": int(valid_ids.min()),
            "max_valid_token_id": int(valid_ids.max()),
            "valid_token_ids_in_range": token_ids_in_range,
            "attention_mask_sum": int(attention.sum()),
        },
        "warning_policy": {
            "observed_issue": "inherited SigLIP config uses out-of-range BOS/EOS defaults",
            "handling": "recorded and corrected before model construction; no warning suppression",
            "weights_changed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
