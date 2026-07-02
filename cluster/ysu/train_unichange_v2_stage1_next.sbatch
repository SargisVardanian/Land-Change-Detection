#!/usr/bin/env bash
#SBATCH --job-name=ucv2-next-train
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
cd "$CODE_ROOT"

MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_stage1_next_train}"
SMOKE_REPORT="${SMOKE_REPORT:?Set SMOKE_REPORT to the successful Stage-1-next smoke report}"
MEMORY_REPORT="${MEMORY_REPORT:?Set MEMORY_REPORT to the successful Stage-1-next memory report}"
CURRENT_COMMIT="$(git rev-parse HEAD)"

BATCH_SIZE="$($PYTHON "$CODE_ROOT/scripts/ucv2_stage1_next_readiness_gate.py" \
  "$SMOKE_REPORT" "$MEMORY_REPORT" "$CURRENT_COMMIT" 1 32)"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
if (( BATCH_SIZE < 32 )); then
  echo "One-H100 Stage-1-next training requires actual batch size at least 32" >&2
  exit 2
fi

mkdir -p "$RS_PROJECT_ROOT/logs" "$RUN_DIR"
ARGS=(
  "$MCI_DATA"
  "$RUN_DIR"
  "$RS_PROJECT_ROOT/external/UniverSat"
  "$RS_PROJECT_ROOT/models/universat-base"
  "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval"
  "$BATCH_SIZE"
  "${EPOCHS:-25}"
  "${NUM_WORKERS:-8}"
  --temporal-depth "${TEMPORAL_DEPTH:-4}"
  --max-captions-per-pair "${MAX_CAPTIONS_PER_PAIR:-2}"
  --caption-frequency-power "${CAPTION_FREQUENCY_POWER:-0.5}"
  --train-eval-pairs "${TRAIN_EVAL_PAIRS:-1024}"
  --train-eval-interval "${TRAIN_EVAL_INTERVAL:-2}"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" "$PYTHON" \
  "$CODE_ROOT/scripts/train_unichange_v2_stage1_next.py" "${ARGS[@]}"
