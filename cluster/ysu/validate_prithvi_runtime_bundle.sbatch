#!/bin/bash
#SBATCH --job-name=rs_prithvi_bundle
#SBATCH --output=/data/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/data/%u/rs_change_project/logs/%x_%j.err
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/runs/prithvi_runtime_bundle.json}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/validate_prithvi_runtime_bundle.py --output "${OUTPUT_PATH}"
