#!/usr/bin/env bash
#SBATCH --job-name=probe_jina_v5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

mkdir -p "$RS_PROJECT_ROOT/reports/model_probes" logs
PYTHONPATH="$RS_PROJECT_ROOT/src" "$PYTHON" "$RS_PROJECT_ROOT/scripts/probe_jina_v5.py" \
  --model-path "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  --output "$RS_PROJECT_ROOT/reports/model_probes/jina_v5.json" \
  --device "${DEVICE:-cpu}"
