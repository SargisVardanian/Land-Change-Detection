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
export LEVIR_MCI_DATA_ROOT="${LEVIR_MCI_DATA_ROOT:-$PROJECT_ROOT/datasets/raw/LEVIR-MCI-unpacked/LEVIR-MCI-dataset}"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/setup_rs_change_project.py --root "$PROJECT_ROOT" --write-manifest
"$PYTHON" scripts/check_datasets.py --root "$PROJECT_ROOT/datasets/raw"
"$PYTHON" scripts/validate_levir_mci_dataset.py \
  --project-root "$PROJECT_ROOT" \
  --data-root "$LEVIR_MCI_DATA_ROOT"
"$PYTHON" scripts/bootstrap_change_retrieval_assets.py --project-root "$PROJECT_ROOT"
"$PYTHON" scripts/render_bootstrap_previews.py --project-root "$PROJECT_ROOT"
"$PYTHON" scripts/render_levir_mci_samples.py \
  --project-root "$PROJECT_ROOT" \
  --data-root "$LEVIR_MCI_DATA_ROOT"

echo "Bootstrap complete."
echo "Indexes:"
echo "  ${PROJECT_ROOT}/indexes"
echo "Previews:"
echo "  ${PROJECT_ROOT}/runs"
