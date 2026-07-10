#!/usr/bin/env bash
#SBATCH --job-name=ucv2-patchseg-train
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
source "${SLURM_SUBMIT_DIR}/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
export ENABLE_PATCH_RERANKER=1
export RUN_DIR="${RUN_DIR:-${RS_PROJECT_ROOT:?}/runs/unichange_v2_patchseg_${SLURM_JOB_ID}}"
bash "${CODE_ROOT}/cluster/ysu/train_unichange_v2_stage1_next.sbatch"
