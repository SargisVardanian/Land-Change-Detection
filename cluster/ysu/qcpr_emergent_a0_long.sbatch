#!/usr/bin/env bash
#SBATCH --job-name=qcpr-a0-long
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_SHA:?}" "${CHAIN_ROOT:?}" "${MANIFEST_DIR:?}"
source "$HOME/rschange_env.sh"
cd "$CODE_ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts"
exec "$RSCHANGE_PYTHON" scripts/train_qcpr_emergent_epochs.py \
  --track A0 --output-dir "$CHAIN_ROOT/a0_long" --manifest-dir "$MANIFEST_DIR" \
  --parent-checkpoint "$CHAIN_ROOT/a0_smoke/last.pt" \
  --baseline-report "$CHAIN_ROOT/a0_smoke/development_step0.json" \
  --prior-run "$CHAIN_ROOT/a0_smoke" --start-epoch 2 --max-epochs 30 --min-epochs 10 --patience 5 \
  --batch-size 32 --learning-rate 5e-5 --text-adapter-learning-rate 5e-6
