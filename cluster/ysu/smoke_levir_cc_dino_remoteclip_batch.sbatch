#!/bin/bash
#SBATCH --job-name=smoke_levir_cc_dino_remoteclip
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

"$PYTHON" scripts/smoke_levir_cc_dino_remoteclip_batch.py \
  --manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries_train.jsonl" \
  --project-root "$RS_PROJECT_ROOT" \
  --output "$RS_PROJECT_ROOT/reports/levir_cc_gpu_smoke.json" \
  --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
  --remoteclip-arch "${REMOTECLIP_ARCH:-ViT-B-32}" \
  --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
  --pair-feature-mode "${PAIR_FEATURE_MODE:-signed_delta}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --device auto
