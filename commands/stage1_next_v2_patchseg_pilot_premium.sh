#!/usr/bin/env bash
#SBATCH --job-name=ucv2-patchseg-pilot
#SBATCH --account=research
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
export MAX_STEPS="${MAX_STEPS:-500}"
export EPOCHS="${EPOCHS:-2}"
# The shape-only probe passed 32, but the first full optimizer-backed pilot
# reached 78 GiB and OOMed. Batch 24 is the single evidence-based fallback.
export MINIMUM_GLOBAL_BATCH="${MINIMUM_GLOBAL_BATCH:-24}"
export BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-24}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_PATCH_RERANKER=1
export QCPR_ARCHITECTURE_VERSION="${QCPR_ARCHITECTURE_VERSION:-v2}"
export ENABLE_TEMPORAL_EXPLANATION_CHANNELS="${ENABLE_TEMPORAL_EXPLANATION_CHANNELS:-1}"
export DATASET_CONFIG="${DATASET_CONFIG:-${RS_PROJECT_ROOT}/configs/qcpr_a78bf8bd/e1_s2looking_masks.json}"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  unset INITIALIZE_FROM_V1
  export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
else
  export INITIALIZE_FROM_V1="${INITIALIZE_FROM_V1:-${RS_PROJECT_ROOT}/runs/qcpr_e0_20260711-015629/pilot/best_retrieval.pt}"
  export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
fi
export RUN_DIR="${RUN_DIR:-${RS_PROJECT_ROOT:?}/runs/unichange_v2_patchseg_pilot_${SLURM_JOB_ID}}"
bash "${CODE_ROOT}/cluster/ysu/train_unichange_v2_stage1_next.sbatch"
