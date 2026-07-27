#!/bin/bash
#SBATCH --job-name=query_pair_to_pair
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
source "$(dirname "$0")/common_env.sh"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/query_pair_to_pair_retrieval.py \
  --manifest "$RS_PROJECT_ROOT/indexes/secondcc_pair_retrieval.jsonl" \
  --checkpoint "$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/best.pt" \
  --query-sample-id "${QUERY_SAMPLE_ID:-sample_0001}" \
  --output "$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/query_topk.png" \
  --visual-backbone simple_patch \
  --local-files-only
