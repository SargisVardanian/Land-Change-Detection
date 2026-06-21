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
export OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/rs_change_project_verification.json}"
source "$(dirname "$0")/common_env.sh"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes"

"$PYTHON" scripts/verify_rs_change_project.py \
  --project-root "$PROJECT_ROOT" \
  --output "$OUTPUT_PATH"

echo "Verification report:"
echo "  ${OUTPUT_PATH}"
