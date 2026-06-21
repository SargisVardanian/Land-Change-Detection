#!/bin/bash
#SBATCH --job-name=train_levir_cc_retrieval
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

PRESET="${PRESET:-simple_patch_smoke}"
RUN_NAME="${RUN_NAME:-$PRESET}"
RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train"

cd "$CODE_ROOT"
mkdir -p logs "$RUN_DIR" "$RS_PROJECT_ROOT/reports"

TRAIN_ARGS=(
  --levir-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries_train.jsonl"
  --eval-levir-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries_val.jsonl"
  --output-dir "$RUN_DIR"
  --project-root "$RS_PROJECT_ROOT"
  --preset "$PRESET"
  --epochs "${EPOCHS:-20}"
  --batch-size "${BATCH_SIZE:-8}"
  --image-size "${IMAGE_SIZE:-224}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}"
  --device auto
)

if [ -n "${LEVIR_PAIR_CACHE_INDEX:-}" ] && [ -n "${LEVIR_TEXT_CACHE_INDEX:-}" ] && [ -f "$LEVIR_PAIR_CACHE_INDEX" ] && [ -f "$LEVIR_TEXT_CACHE_INDEX" ]; then
  TRAIN_ARGS+=(
    --levir-pair-cache-index "$LEVIR_PAIR_CACHE_INDEX"
    --levir-text-cache-index "$LEVIR_TEXT_CACHE_INDEX"
  )
fi

if [ "$PRESET" != "simple_patch_smoke" ]; then
  TRAIN_ARGS+=(
    --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}"
    --remoteclip-arch "${REMOTECLIP_ARCH:-ViT-B-32}"
    --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}"
    --local-files-only
    --mixed-precision
  )
fi

"$PYTHON" scripts/train_dino_pair_retrieval.py "${TRAIN_ARGS[@]}"

"$PYTHON" scripts/summarize_pair_retrieval_train.py \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/train_summary.json"

"$PYTHON" scripts/write_environment_fingerprint.py \
  --project-root "$RS_PROJECT_ROOT" \
  --dataset-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries_train.jsonl" \
  --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
  --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
  --config-json "$RUN_DIR/config.json" \
  --output "$RS_PROJECT_ROOT/reports/environment_fingerprint.json" \
  --cli "scripts/train_dino_pair_retrieval.py ${TRAIN_ARGS[*]}"
