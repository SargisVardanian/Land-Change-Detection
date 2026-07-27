#!/bin/bash
#SBATCH --job-name=audit_levir_cc_milestone
#SBATCH --partition=research
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

PRESET="${PRESET:-simple_patch_smoke}"
RUN_NAME="${RUN_NAME:-$PRESET}"
OUTPUT_PATH="${OUTPUT_PATH:-$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_milestone_audit.json}"

cd "$CODE_ROOT"
mkdir -p logs "$(dirname "$OUTPUT_PATH")"

"$PYTHON" scripts/audit_levir_cc_milestone.py \
  --project-root "$RS_PROJECT_ROOT" \
  --output "$OUTPUT_PATH"
