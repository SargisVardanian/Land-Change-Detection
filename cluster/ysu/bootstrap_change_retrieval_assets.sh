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
python scripts/check_datasets.py --root "${PROJECT_ROOT}/datasets/raw"
python scripts/validate_levir_mci_dataset.py \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${PROJECT_ROOT}/datasets/raw/LEVIR-MCI"
python scripts/bootstrap_change_retrieval_assets.py --project-root "${PROJECT_ROOT}"
python scripts/render_bootstrap_previews.py --project-root "${PROJECT_ROOT}"
python scripts/render_levir_mci_samples.py \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${PROJECT_ROOT}/datasets/raw/LEVIR-MCI"

echo "Bootstrap complete."
echo "Indexes:"
echo "  ${PROJECT_ROOT}/indexes"
echo "Previews:"
echo "  ${PROJECT_ROOT}/runs"
