#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/common_env.sh"

STAGE="${1:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PRESET="${PRESET:-dinov2_change_fusion}"
RUN_NAME="${RUN_NAME:-$PRESET}"

if [ -z "$STAGE" ]; then
  echo "Usage: bash cluster/ysu/run_levir_cc_baseline_end_to_end.sh {preprocess|validate-models|smoke|cache|overfit|train|eval|all}" >&2
  exit 1
fi

submit_stage() {
  local stage_name="$1"
  local dependency="${2:-}"
  shift 2
  local script_path="$1"
  shift
  local sbatch_args=("$SBATCH_BIN" "--parsable")
  if [ -n "$dependency" ]; then
    sbatch_args+=("--dependency=afterok:$dependency")
  fi
  while [ "$#" -gt 0 ]; do
    sbatch_args+=("$1")
    shift
  done
  sbatch_args+=("$script_path")
  local job_id
  job_id="$("${sbatch_args[@]}")"
  echo "$stage_name: $job_id"
  LAST_JOB_ID="$job_id"
}

case "$STAGE" in
  preprocess)
    submit_stage preprocess "" "$SCRIPT_DIR/preprocess_levir_cc.sbatch"
    ;;
  validate-models)
    submit_stage validate-models "" "$SCRIPT_DIR/validate_retrieval_model_assets.sbatch"
    ;;
  smoke)
    submit_stage smoke "" "$SCRIPT_DIR/smoke_levir_cc_dino_remoteclip_batch.sbatch" \
      --export=ALL,PAIR_FEATURE_MODE="${PAIR_FEATURE_MODE:-signed_delta}",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  cache)
    submit_stage cache-dino "" "$SCRIPT_DIR/cache_levir_cc_dinov2_features.sbatch" \
      --export=ALL,DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}"
    cache_dino_id="$LAST_JOB_ID"
    submit_stage cache-text "$cache_dino_id" "$SCRIPT_DIR/cache_levir_cc_remoteclip_text.sbatch" \
      --export=ALL,REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  overfit)
    submit_stage overfit "" "$SCRIPT_DIR/overfit_levir_cc_retrieval_100.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  train)
    submit_stage train "" "$SCRIPT_DIR/train_levir_cc_retrieval.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  eval)
    submit_stage eval "" "$SCRIPT_DIR/eval_levir_cc_retrieval.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train}",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  all)
    submit_stage preprocess "" "$SCRIPT_DIR/preprocess_levir_cc.sbatch"
    preprocess_id="$LAST_JOB_ID"
    submit_stage validate-models "$preprocess_id" "$SCRIPT_DIR/validate_retrieval_model_assets.sbatch" \
      --export=ALL,DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    validate_id="$LAST_JOB_ID"
    submit_stage smoke "$validate_id" "$SCRIPT_DIR/smoke_levir_cc_dino_remoteclip_batch.sbatch" \
      --export=ALL,PAIR_FEATURE_MODE="${PAIR_FEATURE_MODE:-signed_delta}",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    smoke_id="$LAST_JOB_ID"
    submit_stage cache-dino "$smoke_id" "$SCRIPT_DIR/cache_levir_cc_dinov2_features.sbatch" \
      --export=ALL,DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}"
    cache_dino_id="$LAST_JOB_ID"
    submit_stage cache-text "$cache_dino_id" "$SCRIPT_DIR/cache_levir_cc_remoteclip_text.sbatch" \
      --export=ALL,REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    cache_text_id="$LAST_JOB_ID"
    submit_stage overfit "$cache_text_id" "$SCRIPT_DIR/overfit_levir_cc_retrieval_100.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    overfit_id="$LAST_JOB_ID"
    submit_stage train "$overfit_id" "$SCRIPT_DIR/train_levir_cc_retrieval.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    train_id="$LAST_JOB_ID"
    submit_stage eval "$train_id" "$SCRIPT_DIR/eval_levir_cc_retrieval.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    eval_id="$LAST_JOB_ID"
    submit_stage render "$eval_id" "$SCRIPT_DIR/render_levir_cc_text_query_grid.sbatch" \
      --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 1
    ;;
esac
