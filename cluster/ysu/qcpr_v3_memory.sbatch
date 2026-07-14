#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-rc1-memory
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
cd "$ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
for batch in 8 16 20 24; do
  "$PYTHON" scripts/train_qcpr_v3.py --output-dir "$OUTPUT_ROOT/bs${batch}" --phase mask_grounding --steps 3 --batch-size "$batch"
done
