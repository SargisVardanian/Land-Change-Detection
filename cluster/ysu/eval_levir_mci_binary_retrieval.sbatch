#!/bin/bash
#SBATCH --job-name=levir_mci_eval
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
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/eval_levir_mci_binary_retrieval.py \
  --project-root "$RS_PROJECT_ROOT" \
  --data-root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset" \
  --checkpoint "$RS_PROJECT_ROOT/runs/levir_mci_binary_retrieval/best.pt" \
  --output "$RS_PROJECT_ROOT/runs/levir_mci_binary_retrieval/test_metrics.json"
