#!/usr/bin/env bash
#SBATCH --job-name=unichange-mci100
#SBATCH --output=logs/unichange_mci100_%j.out
#SBATCH --error=logs/unichange_mci100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$CODE_ROOT"
RUN_DIR="$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100"
MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/train_unichange_joint.py" \
  --data-root "$MCI_DATA" \
  --subset-file "$RUN_DIR/subset_100.json" \
  --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
  --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
  --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  --output-dir "$RUN_DIR" \
  --max-pairs 100 \
  --batch-size "${BATCH_SIZE:-4}" \
  --epochs "${EPOCHS:-10}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device cuda
