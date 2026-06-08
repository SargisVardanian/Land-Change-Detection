#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/runs/rs_change_project_verification.json}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
  conda activate rschange
else
  source ~/venvs/rschange/bin/activate
fi

python scripts/verify_rs_change_project.py \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_PATH}"

echo "Verification report:"
echo "  ${OUTPUT_PATH}"
