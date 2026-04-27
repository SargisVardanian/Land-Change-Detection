#!/usr/bin/env bash
set -euo pipefail

echo "Applying tuned Ollama environment for Apple Silicon..."
launchctl setenv OLLAMA_NUM_PARALLEL 1
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0

osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
sleep 1
open -a Ollama

cat <<'EOF'
Applied:
  OLLAMA_NUM_PARALLEL=1
  OLLAMA_FLASH_ATTENTION=1
  OLLAMA_KV_CACHE_TYPE=q8_0

Ollama was restarted. If you want a more aggressive low-memory mode, switch:
  launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0
EOF
