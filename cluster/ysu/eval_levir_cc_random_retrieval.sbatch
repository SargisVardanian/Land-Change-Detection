#!/bin/bash
#SBATCH --job-name=eval_levir_cc_random
#SBATCH --partition=research
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

cd "$CODE_ROOT"
mkdir -p logs "$RS_PROJECT_ROOT/runs"

"$PYTHON" scripts/eval_levir_cc_random_retrieval.py \
  --manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries_test.jsonl" \
  --output "$RS_PROJECT_ROOT/runs/levir_cc_random_retrieval_eval.json"
