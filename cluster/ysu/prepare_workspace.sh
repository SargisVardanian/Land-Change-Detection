#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CLUSTER_USER="${LCDC_CLUSTER_USER:-svardanyan}"
PROJECT_SLUG="${LCDC_PROJECT_SLUG:-land-change-detection}"
CONDA_ENV_NAME="${LCDC_CONDA_ENV:-lcd}"

"${PYTHON_BIN}" scripts/prepare_ysu_cluster.py \
  --repo-root "${ROOT}" \
  --cluster-user "${CLUSTER_USER}" \
  --project-slug "${PROJECT_SLUG}" \
  --conda-env-name "${CONDA_ENV_NAME}" \
  --symlink \
  --write-env-file "${ROOT}/cluster/ysu/private/cluster.env"

echo
echo "Workspace prepared."
echo "Next:"
echo "  1. source cluster/ysu/private/cluster.env"
echo "  2. create or activate your Conda env"
echo "  3. submit jobs with sbatch cluster/ysu/train_qwen_lora.sbatch"
