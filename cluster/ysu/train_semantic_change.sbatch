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
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/indexes/semantic/second_cc_change_mask_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/semantic_change_train}"
EPOCHS="${EPOCHS:-3}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/train_semantic_change.py \
  --manifest "${MANIFEST_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}"

python scripts/eval_semantic_change.py \
  --manifest "${MANIFEST_PATH}" \
  --checkpoint "${OUTPUT_DIR}/semantic_change_model.pt" \
  --output "${OUTPUT_DIR}/eval_metrics.json"
