#!/bin/bash
#SBATCH --job-name=rschange_smoke
#SBATCH --output=/data/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/data/%u/rs_change_project/logs/%x_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/runs/cluster_smoke_report.json}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/build_cluster_smoke_report.py \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_PATH}"
