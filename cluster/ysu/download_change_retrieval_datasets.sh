#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/project}"
INCLUDE_RESEARCH_MODELS="${INCLUDE_RESEARCH_MODELS:-0}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/checkpoints/models/semantic}"

cd "${CODE_ROOT}"
source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
  conda activate rschange
else
  source ~/venvs/rschange/bin/activate
fi

python scripts/setup_rs_change_project.py --root "${PROJECT_ROOT}" --write-manifest
python scripts/download_change_retrieval_datasets.py --project-root "${PROJECT_ROOT}" "$@"
if [ "${INCLUDE_RESEARCH_MODELS}" = "1" ]; then
  python scripts/download_semantic_models.py --include-research --output-root "${MODEL_ROOT}"
else
  python scripts/download_semantic_models.py --output-root "${MODEL_ROOT}"
fi

echo "Download stage complete."
echo "Raw datasets:"
echo "  ${PROJECT_ROOT}/datasets/raw"
echo "Models:"
echo "  ${MODEL_ROOT}"
echo "Reference repos:"
echo "  ${PROJECT_ROOT}/code"
