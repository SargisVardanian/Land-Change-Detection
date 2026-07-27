#!/bin/bash
#SBATCH --job-name=levir_mci_validate
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
source "$(dirname "$0")/common_env.sh"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/validate_levir_mci_dataset.py \
  --project-root "$RS_PROJECT_ROOT" \
  --data-root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset"
