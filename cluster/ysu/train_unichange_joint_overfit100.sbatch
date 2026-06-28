#!/usr/bin/env bash
#SBATCH --job-name=unichange-joint-overfit100
#SBATCH --output=logs/unichange_joint_overfit100_%j.out
#SBATCH --error=logs/unichange_joint_overfit100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
mkdir -p "$RS_PROJECT_ROOT/logs"

PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/train_unichange_joint.py" \
  --data-root "$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI" \
  --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
  --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
  --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  --output-dir "$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100" \
  --run-name joint_retrieval_grounding_overfit_100 \
  --max-pairs 100 \
  --batch-size 4 \
  --epochs 10 \
  --device cuda
