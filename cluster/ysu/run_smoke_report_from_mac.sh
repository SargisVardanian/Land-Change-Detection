#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-${1:-}}"
REMOTE_HOST="${REMOTE_HOST:-cluster.ysu.am}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/${REMOTE_USER}/rs_change_project}"
REPORT_LOCAL_PATH="${REPORT_LOCAL_PATH:-./runs_remote/cluster_smoke_report.json}"

if [ -z "${REMOTE_USER}" ]; then
  echo "Usage: REMOTE_USER=<ysu_user> bash cluster/ysu/run_smoke_report_from_mac.sh"
  echo "   or: bash cluster/ysu/run_smoke_report_from_mac.sh <ysu_user>"
  exit 1
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_REPORT_PATH="${PROJECT_ROOT}/runs/cluster_smoke_report.json"
SBATCH_PATH="${PROJECT_ROOT}/code/project/cluster/ysu/smoke_report.sbatch"

echo "Submitting remote smoke report job on ${SSH_TARGET}..."
ssh "${SSH_TARGET}" "sbatch '${SBATCH_PATH}'"

echo "Smoke report job submitted. Pull the report after completion with:"
echo "  rsync -avP '${SSH_TARGET}:${REMOTE_REPORT_PATH}' '${REPORT_LOCAL_PATH}'"
