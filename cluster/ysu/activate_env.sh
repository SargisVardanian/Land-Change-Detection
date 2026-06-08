#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT}/cluster/ysu/private/cluster.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

CONDA_BASE="${LCDC_CONDA_BASE:-/home/svardanyan/miniconda3}"
CONDA_ENV="${LCDC_CONDA_ENV:-lcd}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
    exit 0
  fi
fi

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
  exit 0
fi

echo "No Conda env named ${CONDA_ENV} and no repo .venv found." >&2
echo "Create one first, then rerun the Slurm job." >&2
exit 1
