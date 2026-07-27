#!/bin/bash
#SBATCH --job-name=rs_retrieval_train
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
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/indexes/levir_mci_train_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/retrieval_train}"
EPOCHS="${EPOCHS:-5}"

cd "${CODE_ROOT}"
source "$(dirname "$0")/common_env.sh"

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/run_change_retrieval_train_eval.py \
  --manifest "${MANIFEST_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}"
