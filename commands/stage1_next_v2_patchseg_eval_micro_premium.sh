#!/usr/bin/env bash
#SBATCH --job-name=ucv2-patchseg-eval-micro
#SBATCH --account=research
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
export MAX_QUERIES="${MAX_QUERIES:-12}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
bash "${CODE_ROOT}/cluster/ysu/evaluate_unichange_v2_stage1_next.sbatch"
