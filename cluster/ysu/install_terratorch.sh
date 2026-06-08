#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/cluster/ysu/activate_env.sh"

bash "${ROOT}/scripts/install_terratorch.sh"
