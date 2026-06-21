#!/bin/bash
#SBATCH --job-name=preprocess_levir_cc
#SBATCH --partition=research
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

cd "$CODE_ROOT"
mkdir -p logs "$RS_PROJECT_ROOT/indexes" "$RS_PROJECT_ROOT/reports"

PREPROCESS_ARGS=(
  --root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-CC"
  --project-root "$RS_PROJECT_ROOT"
)
if [ "${FORCE_PREPROCESS:-0}" = "1" ]; then
  PREPROCESS_ARGS+=(--force)
fi
if [ "${VERIFY_ONLY:-0}" = "1" ]; then
  PREPROCESS_ARGS+=(--verify-only)
fi
if [ -n "${MAX_PAIRS:-}" ]; then
  PREPROCESS_ARGS+=(--max-pairs "$MAX_PAIRS")
fi

"$PYTHON" scripts/preprocess_levir_cc.py "${PREPROCESS_ARGS[@]}"
