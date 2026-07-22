#!/usr/bin/env bash
#SBATCH --job-name=qcpr-emergent-report
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_SHA:?}" "${CHAIN_ROOT:?}"
source "$HOME/rschange_env.sh"
cd "$CODE_ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts"
exec "$RSCHANGE_PYTHON" scripts/build_qcpr_emergent_report.py --chain-root "$CHAIN_ROOT"
