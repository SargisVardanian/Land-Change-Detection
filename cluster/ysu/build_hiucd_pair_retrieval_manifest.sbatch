#!/bin/bash
#SBATCH --job-name=hiucd_pair_manifest
#SBATCH --partition=research
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
cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/indexes" "$RS_PROJECT_ROOT/logs"

"$PYTHON" scripts/build_hiucd_pair_retrieval_manifest.py \
  --root "$RS_PROJECT_ROOT/datasets/raw/Hi-UCD" \
  --output "$RS_PROJECT_ROOT/indexes/hiucd_pair_retrieval.jsonl" \
  --diagnostics-output "$RS_PROJECT_ROOT/logs/hiucd_pair_retrieval_diagnostics.json" \
  --num-classes 9
