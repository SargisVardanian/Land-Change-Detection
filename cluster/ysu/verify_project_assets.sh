#!/bin/bash -l
set -euo pipefail

if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
else
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/data/$USER/rs_change_project}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$RS_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/rs_change_project_verification.json}"

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
