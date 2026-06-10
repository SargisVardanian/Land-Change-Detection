#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-${1:-}}"
REMOTE_HOST="${REMOTE_HOST:-cluster.ysu.am}"
TERMINAL_TARGET="${TERMINAL_TARGET:-conda}"
SSH_LOGIN_SUFFIX="${SSH_LOGIN_SUFFIX:-+${TERMINAL_TARGET}}"
PROJECT_ROOT="${PROJECT_ROOT:-}"
REPORT_LOCAL_PATH="${REPORT_LOCAL_PATH:-./runs_remote/rs_change_project_verification.json}"

if [ -z "${REMOTE_USER}" ]; then
  echo "Usage: REMOTE_USER=<ysu_user> bash cluster/ysu/verify_project_assets_from_mac.sh"
  echo "   or: bash cluster/ysu/verify_project_assets_from_mac.sh <ysu_user>"
  exit 1
fi

SSH_TARGET="${REMOTE_USER}${SSH_LOGIN_SUFFIX}@${REMOTE_HOST}"
if [ -z "${PROJECT_ROOT}" ]; then
  PROJECT_ROOT="$(ssh "${SSH_TARGET}" 'if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then echo "/mnt/weka/$USER/rs_change_project"; else echo "/data/$USER/rs_change_project"; fi')"
fi
REMOTE_REPORT_PATH="${PROJECT_ROOT}/rs_change_project_verification.json"

echo "Running remote asset verification on ${SSH_TARGET}..."
ssh "${SSH_TARGET}" "cd '${PROJECT_ROOT}/code/project' && RS_PROJECT_ROOT='${PROJECT_ROOT}' PROJECT_ROOT='${PROJECT_ROOT}' OUTPUT_PATH='${REMOTE_REPORT_PATH}' bash cluster/ysu/verify_project_assets.sh"

mkdir -p "$(dirname "${REPORT_LOCAL_PATH}")"
rsync -avP "${SSH_TARGET}:${REMOTE_REPORT_PATH}" "${REPORT_LOCAL_PATH}"

echo "Local report:"
echo "  ${REPORT_LOCAL_PATH}"
