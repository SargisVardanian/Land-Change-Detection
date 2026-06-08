#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/artifacts/models/semantic/Prithvi-EO-2.0-300M-TL/terratorch_config.yaml}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate rschange
else
    source ~/venvs/rschange/bin/activate
fi

python scripts/write_prithvi_terratorch_template.py --output "${OUTPUT_PATH}"
