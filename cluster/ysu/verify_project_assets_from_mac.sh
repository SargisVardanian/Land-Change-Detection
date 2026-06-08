#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-${1:-}}"
REMOTE_HOST="${REMOTE_HOST:-cluster.ysu.am}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/${REMOTE_USER}/rs_change_project}"
REPORT_LOCAL_PATH="${REPORT_LOCAL_PATH:-./runs_remote/rs_change_project_verification.json}"

if [ -z "${REMOTE_USER}" ]; then
  echo "Usage: REMOTE_USER=<ysu_user> bash cluster/ysu/verify_project_assets_from_mac.sh"
  echo "   or: bash cluster/ysu/verify_project_assets_from_mac.sh <ysu_user>"
  exit 1
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_REPORT_PATH="${PROJECT_ROOT}/runs/rs_change_project_verification.json"

echo "Running remote asset verification on ${SSH_TARGET}..."
ssh "${SSH_TARGET}" "cd '${PROJECT_ROOT}/code/project' && PROJECT_ROOT='${PROJECT_ROOT}' OUTPUT_PATH='${REMOTE_REPORT_PATH}' bash cluster/ysu/verify_project_assets.sh"

mkdir -p "$(dirname "${REPORT_LOCAL_PATH}")"
rsync -avP "${SSH_TARGET}:${REMOTE_REPORT_PATH}" "${REPORT_LOCAL_PATH}"

echo "Local report:"
echo "  ${REPORT_LOCAL_PATH}"
