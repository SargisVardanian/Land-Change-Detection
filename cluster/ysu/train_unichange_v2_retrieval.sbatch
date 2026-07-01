#!/usr/bin/env bash
#SBATCH --job-name=ucv2-ret-train
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=24:00:00

set -euo pipefail

source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
cd "$CODE_ROOT"

MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_train}"

PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/train_unichange_v2_retrieval.py" \
  --data-root "$MCI_DATA" \
  --output-dir "$RUN_DIR" \
  --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
  --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
  --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  --device cuda \
  --image-size 256 \
  --output-grid 32 \
  --batch-size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-25}" \
  --num-workers "${NUM_WORKERS:-8}"
