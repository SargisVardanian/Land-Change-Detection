#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"${PYTHON_BIN}" -m venv "${ROOT}/.venv"
source "${ROOT}/.venv/bin/activate"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.95}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.85}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT}[dev]"

if [[ "${INSTALL_LLM_EXTRAS:-0}" == "1" ]]; then
  python -m pip install -e "${ROOT}[llm]"
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())
import os
print("PYTORCH_MPS_HIGH_WATERMARK_RATIO:", os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"))
print("PYTORCH_MPS_LOW_WATERMARK_RATIO:", os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"))
print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
PY
