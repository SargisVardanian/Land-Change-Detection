#!/bin/bash
set -euo pipefail

if [ -z "${RS_PROJECT_ROOT:-}" ]; then
  if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
    export RS_PROJECT_ROOT="/mnt/weka/$USER/rs_change_project"
  else
    export RS_PROJECT_ROOT="/data/$USER/rs_change_project"
  fi
fi

export CODE_ROOT="${CODE_ROOT:-$RS_PROJECT_ROOT/code/project}"
export PROJECT_DIR="${PROJECT_DIR:-$CODE_ROOT}"
export PYTHON="${RSCHANGE_PYTHON:-$RS_PROJECT_ROOT/envs/rschange/bin/python}"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RS_PROJECT_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RS_PROJECT_ROOT/.cache/pip}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
