#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-${1:-}}"
REMOTE_USER="${REMOTE_USER:-${2:-}}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/${REMOTE_USER}/rs_change_project}"
REMOTE_CODE_ROOT="${REMOTE_CODE_ROOT:-${PROJECT_ROOT}/code/project}"
INCLUDE_RESEARCH_MODELS="${INCLUDE_RESEARCH_MODELS:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
SETUP_ENV="${SETUP_ENV:-1}"
ENV_NAME="${ENV_NAME:-rschange}"

if [ -z "${REMOTE_HOST}" ] || [ -z "${REMOTE_USER}" ]; then
  echo "Usage: REMOTE_HOST=cluster.ysu.am REMOTE_USER=<ysu_user> bash cluster/ysu/push_and_bootstrap_from_mac.sh"
  echo "   or: bash cluster/ysu/push_and_bootstrap_from_mac.sh cluster.ysu.am <ysu_user>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

echo "Checking SSH reachability for ${SSH_TARGET}..."
ssh -o ConnectTimeout=10 "${SSH_TARGET}" "hostname; whoami"

echo "Creating remote project directories..."
ssh "${SSH_TARGET}" "mkdir -p '${PROJECT_ROOT}/datasets/raw' '${PROJECT_ROOT}/datasets/processed' '${PROJECT_ROOT}/datasets/cache' '${PROJECT_ROOT}/code' '${PROJECT_ROOT}/logs' '${PROJECT_ROOT}/runs' '${PROJECT_ROOT}/checkpoints' '${PROJECT_ROOT}/indexes'"

echo "Syncing project code to ${REMOTE_CODE_ROOT}..."
rsync -avP \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'datasets' \
  --exclude 'checkpoints' \
  --exclude 'runs_remote' \
  "${REPO_ROOT}/" "${SSH_TARGET}:${REMOTE_CODE_ROOT}/"

if [ "${SETUP_ENV}" = "1" ]; then
  echo "Preparing remote Python environment..."
  ssh "${SSH_TARGET}" "cd '${REMOTE_CODE_ROOT}' && PROJECT_ROOT='${PROJECT_ROOT}' ENV_NAME='${ENV_NAME}' bash cluster/ysu/setup_rschange_env.sh"
fi

echo "Launching remote dataset/model download..."
ssh "${SSH_TARGET}" "cd '${REMOTE_CODE_ROOT}' && PROJECT_ROOT='${PROJECT_ROOT}' INCLUDE_RESEARCH_MODELS='${INCLUDE_RESEARCH_MODELS}' bash cluster/ysu/download_change_retrieval_datasets.sh"

if [ "${RUN_BOOTSTRAP}" = "1" ]; then
  echo "Launching remote bootstrap/index build..."
  ssh "${SSH_TARGET}" "cd '${REMOTE_CODE_ROOT}' && PROJECT_ROOT='${PROJECT_ROOT}' bash cluster/ysu/bootstrap_change_retrieval_assets.sh"
fi

echo "Remote setup complete."
echo "Project root: ${PROJECT_ROOT}"
echo "Code root: ${REMOTE_CODE_ROOT}"
echo "Datasets: ${PROJECT_ROOT}/datasets/raw"
echo "Models: ${PROJECT_ROOT}/checkpoints/models/semantic"
