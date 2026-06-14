#!/bin/bash
#SBATCH --job-name=eval_dino_pair_retrieval
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/eval_dino_pair_retrieval.py \
  --levir-manifest "$RS_PROJECT_ROOT/indexes/levir_mci_samples.jsonl" \
  --pair-manifest "$RS_PROJECT_ROOT/indexes/secondcc_pair_retrieval.jsonl" \
  --checkpoint "$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/best.pt" \
  --output "$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/eval_metrics.json" \
  --visual-backbone simple_patch \
  --local-files-only
