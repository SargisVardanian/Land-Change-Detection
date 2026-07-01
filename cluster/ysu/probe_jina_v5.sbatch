#!/usr/bin/env bash
#SBATCH --job-name=probe_jina_v5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/cluster/ysu/common_env.sh"
cd "${CODE_ROOT}"

mkdir -p "${RS_PROJECT_ROOT}/reports/model_probes" logs
PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}" "${PYTHON}" "${CODE_ROOT}/scripts/probe_jina_v5.py" \
  --model-path "${RS_PROJECT_ROOT}/models/jina-v5-text-small-retrieval" \
  --output "${RS_PROJECT_ROOT}/reports/model_probes/jina_v5.json" \
  --device "${DEVICE:-cpu}"
