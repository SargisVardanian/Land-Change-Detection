#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/.venv/bin/activate"

MODEL_PATH="${1:-artifacts/models/mlx-community__Qwen3.5-0.8B-OptiQ-4bit}"
DATA_PATH="${2:-artifacts/qwen_lora/data}"
ADAPTER_PATH="${3:-artifacts/qwen_lora/adapters}"

python scripts/build_qwen_lora_dataset.py

python -m mlx_lm lora \
  --model "${MODEL_PATH}" \
  --train \
  --data "${DATA_PATH}" \
  --iters 200 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --mask-prompt \
  --adapter-path "${ADAPTER_PATH}"
