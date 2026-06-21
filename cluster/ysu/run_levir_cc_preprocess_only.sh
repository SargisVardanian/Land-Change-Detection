#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"

job_id="$("$SBATCH_BIN" --parsable "$SCRIPT_DIR/preprocess_levir_cc.sbatch")"
echo "preprocess: $job_id"
