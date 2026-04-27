#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/.venv/bin/activate"

python -m pip install -U "mlx-lm>=0.30.7" "mlx-vlm>=0.3.12" huggingface_hub

python - <<'PY'
import importlib.util
for name in ["mlx", "mlx_lm", "mlx_vlm", "huggingface_hub"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
