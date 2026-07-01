#!/usr/bin/env bash
#SBATCH --job-name=ucv2-ret-2gpu
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
cd "$CODE_ROOT"
MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_train_2gpu}"
SMOKE_REPORT="${SMOKE_REPORT:?missing SMOKE_REPORT}"
MEMORY_REPORT="${MEMORY_REPORT:?missing MEMORY_REPORT}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
LOCAL_BATCH="$($PYTHON "$CODE_ROOT/scripts/ucv2_readiness_gate.py" "$SMOKE_REPORT" "$MEMORY_REPORT" "$CURRENT_COMMIT" 2 32)"
LOCAL_BATCH="${LOCAL_BATCH_OVERRIDE:-$LOCAL_BATCH}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
mkdir -p "$RS_PROJECT_ROOT/logs" "$RUN_DIR"
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 "$CODE_ROOT/scripts/train_unichange_v2_retrieval_ddp.py" "$MCI_DATA" "$RUN_DIR" "$RS_PROJECT_ROOT/external/UniverSat" "$RS_PROJECT_ROOT/models/universat-base" "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" "$LOCAL_BATCH" "${EPOCHS:-25}" "${NUM_WORKERS:-8}" "$RESUME_CHECKPOINT"
