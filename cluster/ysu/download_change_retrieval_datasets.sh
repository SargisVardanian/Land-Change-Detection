#!/bin/bash -l
set -euo pipefail

if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
else
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/data/$USER/rs_change_project}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$RS_PROJECT_ROOT}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
  conda activate rschange
else
  source ~/venvs/rschange/bin/activate
fi

python scripts/setup_rs_change_project.py --root "${PROJECT_ROOT}" --write-manifest
python scripts/download_change_retrieval_datasets.py --project-root "${PROJECT_ROOT}" --skip-second-cc --skip-reference-repos "$@"

echo "Download stage complete."
echo "Raw datasets:"
echo "  ${PROJECT_ROOT}/datasets/raw"
