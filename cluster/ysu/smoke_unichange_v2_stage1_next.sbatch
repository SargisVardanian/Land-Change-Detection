#!/usr/bin/env bash
#SBATCH --job-name=ucv2-next-smoke
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
cd "$CODE_ROOT"

MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_stage1_next_smoke_${SLURM_JOB_ID}}"
TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-6}"
mkdir -p "$RS_PROJECT_ROOT/logs" "$RUN_DIR"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" "$PYTHON" \
  "$CODE_ROOT/scripts/smoke_unichange_v2_stage1_next.py" \
  "$MCI_DATA" "$RUN_DIR" \
  "$RS_PROJECT_ROOT/external/UniverSat" \
  "$RS_PROJECT_ROOT/models/universat-base" \
  "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  "$TEMPORAL_DEPTH"
