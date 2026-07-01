#!/usr/bin/env bash
#SBATCH --job-name=ucv2-ret-eval
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/cluster/ysu/common_env.sh"
cd "${CODE_ROOT}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed training directory}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best_retrieval.pt}"
MANIFEST="${MANIFEST:-${RS_PROJECT_ROOT}/reports/unichange_v2_eval_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation}"
MCI_DATA="${MCI_ROOT:-${RS_PROJECT_ROOT}/datasets/processed/LEVIR-MCI}"

mkdir -p "${RS_PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
test -f "${CHECKPOINT}"
test -f "${MANIFEST}"

PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}/scripts:${CODE_ROOT}" "${PYTHON}" "${CODE_ROOT}/scripts/render_unichange_v2_retrieval.py" \
  --data-root "${MCI_DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest "${MANIFEST}" \
  --universat-source "${RS_PROJECT_ROOT}/external/UniverSat" \
  --universat-checkpoint "${RS_PROJECT_ROOT}/models/universat-base" \
  --jina-model "${RS_PROJECT_ROOT}/models/jina-v5-text-small-retrieval" \
  --split "${SPLIT:-val}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --top-k "${TOP_K:-5}" \
  --max-queries "${MAX_QUERIES:-48}" \
  --device cuda

PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}/scripts:${CODE_ROOT}" "${PYTHON}" "${CODE_ROOT}/scripts/plot_unichange_v2_training.py" \
  --history "${RUN_DIR}/metrics_history.jsonl" \
  --output-dir "${RUN_DIR}"
