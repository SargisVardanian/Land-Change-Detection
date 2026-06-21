#!/bin/bash
#SBATCH --job-name=rs_prithvi_manifest
#SBATCH --output=/data/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/data/%u/rs_change_project/logs/%x_%j.err
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${PROJECT_ROOT}/indexes/semantic/second_cc_transition_manifest.jsonl}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${PROJECT_ROOT}/indexes/semantic/second_cc_prithvi_manifest.jsonl}"

cd "${CODE_ROOT}"
source "$(dirname "$0")/common_env.sh"

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/build_prithvi_semantic_manifest.py \
  --input-manifest "${INPUT_MANIFEST}" \
  --output "${OUTPUT_MANIFEST}"
