#!/bin/bash
#SBATCH --job-name=dinov2_smoke_clean
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
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/models"

"$PYTHON" scripts/smoke_dinov2_local.py \
  --model-path "$RS_PROJECT_ROOT/models/dinov2-small" \
  --device auto
