#!/bin/bash
#SBATCH --job-name=lcd_qwen_lora
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/weka/svardanyan/land-change-detection/slurm_logs/qwen_lora_%j.out

set -euo pipefail

ROOT="${ROOT:-$HOME/Land-Change-Detection}"
cd "${ROOT}"

source cluster/ysu/activate_env.sh

export HF_HOME="${HF_HOME:-/mnt/weka/svardanyan/land-change-detection/artifacts/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-artifacts/models/mlx-community__Qwen3.5-0.8B-OptiQ-4bit}"
DATA_PATH="${DATA_PATH:-artifacts/qwen_lora/data}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/qwen_lora/adapters}"

bash scripts/train_qwen_lora.sh "${MODEL_PATH}" "${DATA_PATH}" "${ADAPTER_PATH}"
