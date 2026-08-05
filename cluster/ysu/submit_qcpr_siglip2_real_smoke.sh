#!/usr/bin/env bash
#SBATCH --job-name=qcpr_siglip2_smoke
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --account=research
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -eu
ROOT=${PROJECT_ROOT:-/mnt/weka/svardanyan/rs_change_project}
WT=${WORKTREE:-$ROOT/code/project-qcpr-siglip2-temporal-training}
PY=${PYTHON:-$ROOT/envs/rschange/bin/python}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
DATA_RELEASE=${DATA_RELEASE:?set DATA_RELEASE}
SIGLIP2_MODEL=${SIGLIP2_MODEL:?set SIGLIP2_MODEL}
TRAIN_MANIFEST=${TRAIN_MANIFEST:?set TRAIN_MANIFEST}
STEPS=${STEPS:-8}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-8}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
CONFIG_PATH=${CONFIG_PATH:-$WT/configs/qcpr_siglip2_phase_a.json}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-none}

test -d "$WT"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$CONFIG_PATH"
test -d "$DATA_RELEASE"
test -s "$TRAIN_MANIFEST"
test -d "$SIGLIP2_MODEL"
test -f "$SIGLIP2_MODEL/model.safetensors"
test -f "$SIGLIP2_MODEL/config.json"
if test "$CHECKPOINT_PATH" != "none"; then test -s "$CHECKPOINT_PATH"; fi

export HF_HUB_OFFLINE=1
export PYTHONPATH="$WT/src:$WT/scripts"
export PYTHONWARNINGS=default
exec "$PY" "$WT/scripts/run_qcpr_siglip2_real_smoke.py"   --siglip2-model "$SIGLIP2_MODEL"   --train-manifest "$TRAIN_MANIFEST"   --output-dir "$RUN_ROOT"   --expected-code-sha "$EXPECTED_SHA"   --steps "$STEPS"   --physical-batch-size "$PHYSICAL_BATCH_SIZE"   --captions-per-pair "$CAPTIONS_PER_PAIR"
