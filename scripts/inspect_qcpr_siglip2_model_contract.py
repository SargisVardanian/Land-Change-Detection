#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from transformers import AutoModel

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, dtype="float32", low_cpu_mem_usage=True).eval()
    config = model.config
    report = {
        "runtime_class": type(model).__name__,
        "model_type": config.model_type,
        "vision_hidden_size": config.vision_config.hidden_size,
        "vision_patch_size": config.vision_config.patch_size,
        "vision_image_size": config.vision_config.image_size,
        "vision_layers": config.vision_config.num_hidden_layers,
        "text_hidden_size": config.text_config.hidden_size,
        "text_layers": config.text_config.num_hidden_layers,
        "text_max_position_embeddings": config.text_config.max_position_embeddings,
        "vision_parameter_count": sum(p.numel() for p in model.vision_model.parameters()),
        "text_parameter_count": sum(p.numel() for p in model.text_model.parameters()),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
