#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cluster.ysu.am}"
SSH_ALIAS="${SSH_ALIAS:-ysu-hpc}"
OUTPUT_PATH="${OUTPUT_PATH:-./runs_remote/ysu_access_diagnostics.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/diagnose_ysu_access.py" \
  --host "${REMOTE_HOST}" \
  --ssh-alias "${SSH_ALIAS}" \
  --wireguard-dir "${REPO_ROOT}/cluster/ysu/private/wireguard" \
  --output "${OUTPUT_PATH}"

echo "Access diagnostics written to:"
echo "  ${OUTPUT_PATH}"
