#!/bin/bash
#SBATCH --job-name=validate_levir_cc_pair_manifest
#SBATCH --partition=research
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes"

"$PYTHON" scripts/validate_levir_cc_pair_manifest.py \
  --manifest "$RS_PROJECT_ROOT/indexes/levir_cc_pair_manifest.jsonl" \
  --output "$RS_PROJECT_ROOT/runs/levir_cc_manifest_validation.json"
