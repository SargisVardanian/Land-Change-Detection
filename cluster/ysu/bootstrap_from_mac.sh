#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-${1:-}}"
REMOTE_HOST="${REMOTE_HOST:-cluster.ysu.am}"
SSH_ALIAS="${SSH_ALIAS:-ysu-hpc}"
WRITE_SSH_CONFIG="${WRITE_SSH_CONFIG:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${REMOTE_USER}" ]; then
  echo "Usage: REMOTE_USER=<ysu_user> bash cluster/ysu/bootstrap_from_mac.sh"
  echo "   or: bash cluster/ysu/bootstrap_from_mac.sh <ysu_user>"
  exit 1
fi

if [ "${WRITE_SSH_CONFIG}" = "1" ]; then
  echo "Writing SSH config entry for ${SSH_ALIAS}..."
  PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/write_ysu_ssh_config.py" \
    --alias "${SSH_ALIAS}" \
    --host "${REMOTE_HOST}" \
    --user "${REMOTE_USER}"
fi

echo "Running YSU-HPC network/access diagnostics..."
set +e
bash "${REPO_ROOT}/cluster/ysu/diagnose_access_from_mac.sh"
ACCESS_STATUS=$?
set -e

if [ "${ACCESS_STATUS}" -ne 0 ]; then
  echo "Access diagnostics failed. Review runs_remote/ysu_access_diagnostics.json before bootstrap."
fi

echo "Running YSU-HPC preflight..."
set +e
PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/ysu_cluster_preflight.py" \
  --host "${REMOTE_HOST}" \
  --ssh-alias "${SSH_ALIAS}"
PREFLIGHT_STATUS=$?
set -e

if [ "${PREFLIGHT_STATUS}" -ne 0 ]; then
  echo "Preflight failed. Fix VPN/DNS/SSH first, then rerun this script."
  exit "${PREFLIGHT_STATUS}"
fi

echo "Preflight passed. Launching remote bootstrap..."
REMOTE_HOST="${REMOTE_HOST}" REMOTE_USER="${REMOTE_USER}" \
  bash "${REPO_ROOT}/cluster/ysu/push_and_bootstrap_from_mac.sh"

echo "Collecting post-bootstrap verification report..."
REMOTE_HOST="${REMOTE_HOST}" REMOTE_USER="${REMOTE_USER}" \
  bash "${REPO_ROOT}/cluster/ysu/verify_project_assets_from_mac.sh"
