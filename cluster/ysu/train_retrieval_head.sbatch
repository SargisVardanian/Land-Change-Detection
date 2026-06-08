#!/bin/bash
#SBATCH --job-name=lcd_retrieval
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/mnt/weka/svardanyan/land-change-detection/slurm_logs/retrieval_%j.out

set -euo pipefail

ROOT="${ROOT:-$HOME/Land-Change-Detection}"
cd "${ROOT}"

source cluster/ysu/activate_env.sh

MANIFEST_PATH="${MANIFEST_PATH:-data/retrieval/train_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/retrieval/train_run}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_DIR}/eval_metrics.json}"
EPOCHS="${EPOCHS:-5}"

python scripts/train_retrieval_head.py \
  --manifest "${MANIFEST_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}"

python scripts/eval_retrieval.py \
  --manifest "${MANIFEST_PATH}" \
  --output "${EVAL_OUTPUT}"
