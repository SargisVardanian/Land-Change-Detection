#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_MODE="${INSTALL_MODE:-editable}"
TERRATORCH_SPEC="${TERRATORCH_SPEC:-terratorch}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  :
else
  echo "No active environment found. Activate Conda or create .venv first." >&2
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_MODE}" == "editable" ]]; then
  python -m pip install -e "${ROOT}[dev,eo]"
else
  python -m pip install "${ROOT}[dev,eo]"
fi

python -m pip install "${TERRATORCH_SPEC}"

python scripts/diagnose_terratorch_env.py --output "${ROOT}/artifacts/terratorch_env_diagnostics.json"
echo "TerraTorch install flow complete."
