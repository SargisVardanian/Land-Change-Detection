#!/bin/bash
set -euo pipefail
set -x

if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
else
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/data/$USER/rs_change_project}"
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$RS_PROJECT_ROOT}"
export PROJECT_DIR="${PROJECT_DIR:-${PROJECT_ROOT}/code/project}"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/setup_rs_change_project.py --root "$PROJECT_ROOT" --write-manifest
"$PYTHON" scripts/download_change_retrieval_datasets.py --project-root "$PROJECT_ROOT" --skip-second-cc --skip-reference-repos "$@"

echo "Download stage complete."
echo "Raw datasets:"
echo "  ${PROJECT_ROOT}/datasets/raw"
