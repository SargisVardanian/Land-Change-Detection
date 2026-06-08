#!/bin/bash
#SBATCH --job-name=rs_semantic_manifest
#SBATCH --output=/data/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/data/%u/rs_change_project/logs/%x_%j.err
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

if [ -f "${PROJECT_ROOT}/indexes/levir_mci_samples.jsonl" ]; then
  python scripts/build_semantic_change_task_manifests.py \
    --index "${PROJECT_ROOT}/indexes/levir_mci_samples.jsonl" \
    --dataset-name LEVIR-MCI \
    --output-dir "${PROJECT_ROOT}/indexes/semantic"
fi

if [ -f "${PROJECT_ROOT}/indexes/second_cc_samples.jsonl" ]; then
  python scripts/build_semantic_change_task_manifests.py \
    --index "${PROJECT_ROOT}/indexes/second_cc_samples.jsonl" \
    --dataset-name SECOND-CC \
    --output-dir "${PROJECT_ROOT}/indexes/semantic"
fi
