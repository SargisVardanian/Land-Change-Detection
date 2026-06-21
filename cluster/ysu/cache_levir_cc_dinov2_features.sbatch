#!/bin/bash
#SBATCH --job-name=cache_levir_cc_dinov2
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

cd "$CODE_ROOT"
mkdir -p logs "$RS_PROJECT_ROOT/cache/levir_cc_dinov2"

"$PYTHON" scripts/cache_levir_cc_dinov2_features.py \
  --pair-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_pairs.jsonl" \
  --output-dir "$RS_PROJECT_ROOT/cache/levir_cc_dinov2" \
  --project-root "$RS_PROJECT_ROOT" \
  --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --device auto
