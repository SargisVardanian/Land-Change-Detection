#!/bin/bash
#SBATCH --job-name=validate_retrieval_models
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

cd "$CODE_ROOT"
mkdir -p logs "$RS_PROJECT_ROOT/reports"

"$PYTHON" scripts/validate_retrieval_model_assets.py \
  --project-root "$RS_PROJECT_ROOT" \
  --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
  --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
  --remoteclip-arch "${REMOTECLIP_ARCH:-ViT-B-32}" \
  --output "$RS_PROJECT_ROOT/reports/retrieval_model_assets.json"
