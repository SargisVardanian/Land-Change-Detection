#!/bin/bash
#SBATCH --job-name=rs_semantic_train
#SBATCH --output=/data/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/data/%u/rs_change_project/logs/%x_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
TRAIN_MANIFEST_PATH="${TRAIN_MANIFEST_PATH:-${PROJECT_ROOT}/indexes/semantic/second_cc_transition_train_manifest.jsonl}"
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-${PROJECT_ROOT}/indexes/semantic/second_cc_transition_val_manifest.jsonl}"
TEST_MANIFEST_PATH="${TEST_MANIFEST_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/semantic_change_train}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cd "${CODE_ROOT}"
source "$(dirname "$0")/common_env.sh"

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

EXTRA_ARGS=()
if [[ -n "${TEST_MANIFEST_PATH}" ]]; then
  EXTRA_ARGS+=(--test-manifest "${TEST_MANIFEST_PATH}")
fi

python scripts/train_semantic_change.py \
  --train-manifest "${TRAIN_MANIFEST_PATH}" \
  --val-manifest "${VAL_MANIFEST_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

python scripts/eval_semantic_change.py \
  --manifest "${VAL_MANIFEST_PATH}" \
  --checkpoint "${OUTPUT_DIR}/best.pt" \
  --output "${OUTPUT_DIR}/eval_metrics.json" \
  --batch-size "${BATCH_SIZE}"
