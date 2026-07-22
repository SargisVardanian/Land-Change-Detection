#!/usr/bin/env bash
#SBATCH --job-name=qcpr-c0-emergent-eval
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_SHA:?}" "${CHAIN_ROOT:?}" "${S2LOOKING_MANIFEST_DIR:?}"
source "$HOME/rschange_env.sh"
cd "$CODE_ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts"
exec "$RSCHANGE_PYTHON" scripts/evaluate_qcpr_c0_emergent.py \
  --checkpoint "$CHAIN_ROOT/b_long/best_feasible.pt" \
  --manifest "$S2LOOKING_MANIFEST_DIR/m0_dev_manifest.jsonl" \
  --output-dir "$CHAIN_ROOT/c0_development"
