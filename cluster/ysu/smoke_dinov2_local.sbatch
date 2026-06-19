#!/bin/bash
#SBATCH --job-name=smoke_dinov2_local
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/models" "$HF_HOME"

"$PYTHON" scripts/smoke_dinov2_local.py \
  --model-path "$RS_PROJECT_ROOT/models/dinov2-small" \
  --device auto
