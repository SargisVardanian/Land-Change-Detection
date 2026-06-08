#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
ENV_NAME="${ENV_NAME:-rschange}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi
  conda activate "${ENV_NAME}"
else
  VENV_PATH="${HOME}/venvs/${ENV_NAME}"
  python3 -m venv "${VENV_PATH}"
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${CODE_ROOT}[dev]"
python -m pip install -U hf_xet datasets gdown zenodo-get tqdm pandas pillow opencv-python pyarrow fastparquet

echo "Environment ready."
echo "Code root: ${CODE_ROOT}"
echo "Active python: $(command -v python)"
