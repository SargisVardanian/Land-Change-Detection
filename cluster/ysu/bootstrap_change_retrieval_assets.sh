#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/$USER/rs_change_project}"
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
python scripts/bootstrap_change_retrieval_assets.py --project-root "${PROJECT_ROOT}"
python scripts/render_bootstrap_previews.py --project-root "${PROJECT_ROOT}"
if [ -f "${PROJECT_ROOT}/indexes/levir_mci_samples.jsonl" ]; then
  python scripts/build_change_retrieval_training_manifest.py \
    --index "${PROJECT_ROOT}/indexes/levir_mci_samples.jsonl" \
    --output "${PROJECT_ROOT}/indexes/levir_mci_train_manifest.jsonl"
  python scripts/build_semantic_change_task_manifests.py \
    --index "${PROJECT_ROOT}/indexes/levir_mci_samples.jsonl" \
    --dataset-name LEVIR-MCI \
    --output-dir "${PROJECT_ROOT}/indexes/semantic"
fi
if [ -f "${PROJECT_ROOT}/indexes/second_cc_samples.jsonl" ]; then
  python scripts/build_change_retrieval_training_manifest.py \
    --index "${PROJECT_ROOT}/indexes/second_cc_samples.jsonl" \
    --output "${PROJECT_ROOT}/indexes/second_cc_train_manifest.jsonl"
  python scripts/build_semantic_change_task_manifests.py \
    --index "${PROJECT_ROOT}/indexes/second_cc_samples.jsonl" \
    --dataset-name SECOND-CC \
    --output-dir "${PROJECT_ROOT}/indexes/semantic"
fi
if [ -f "${PROJECT_ROOT}/indexes/levir_cc_text_manifest.jsonl" ]; then
  python scripts/build_levir_cc_text_index.py \
    --manifest "${PROJECT_ROOT}/indexes/levir_cc_text_manifest.jsonl" \
    --output-dir "${PROJECT_ROOT}/indexes/levir_cc_text_index"
fi

echo "Bootstrap complete."
echo "Indexes:"
echo "  ${PROJECT_ROOT}/indexes"
echo "Previews:"
echo "  ${PROJECT_ROOT}/runs"
