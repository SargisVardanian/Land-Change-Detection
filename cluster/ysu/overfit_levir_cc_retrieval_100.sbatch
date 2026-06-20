#!/bin/bash
#SBATCH --job-name=overfit_levir_cc_retrieval_100
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"

PRESET="${PRESET:-simple_patch_smoke}"
RUN_NAME="${RUN_NAME:-$PRESET}"
RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME" "$RUN_DIR"

TRAIN_ARGS=(
  --levir-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_pair_manifest.jsonl"
  --output-dir "$RUN_DIR"
  --project-root "$RS_PROJECT_ROOT"
  --preset "$PRESET"
  --epochs "${EPOCHS:-12}"
  --batch-size "${BATCH_SIZE:-8}"
  --image-size "${IMAGE_SIZE:-224}"
  --max-train-samples "${MAX_TRAIN_SAMPLES:-100}"
  --device auto
)

if [ "$PRESET" != "simple_patch_smoke" ]; then
  TRAIN_ARGS+=(--dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}")
  TRAIN_ARGS+=(--remoteclip-model-path "${REMOTECLIP_MODEL_PATH:-$RS_PROJECT_ROOT/models/remoteclip-rn50}")
  TRAIN_ARGS+=(--local-files-only)
fi

"$PYTHON" scripts/overfit_levir_cc_retrieval_100.py "${TRAIN_ARGS[@]}"

"$PYTHON" scripts/summarize_pair_retrieval_train.py \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/train_summary.json"
