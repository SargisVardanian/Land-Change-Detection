#!/usr/bin/env bash
#SBATCH --job-name=probe_universat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$CODE_ROOT"

mkdir -p "$RS_PROJECT_ROOT/reports/model_probes" logs
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/probe_universat.py" \
  --source-dir "$RS_PROJECT_ROOT/external/UniverSat" \
  --checkpoint-dir "$RS_PROJECT_ROOT/models/universat-base" \
  --output "$RS_PROJECT_ROOT/reports/model_probes/universat.json" \
  --device "${DEVICE:-cuda}"
