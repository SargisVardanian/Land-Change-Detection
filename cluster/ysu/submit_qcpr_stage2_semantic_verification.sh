#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project
WORKTREE=$ROOT/code/project-qcpr-dataset-v2-stage2
PYTHON=$ROOT/envs/rschange/bin/python
AUDIT=$ROOT/manifests/qcpr_dataset_v2_stage2_audit
MODEL=$ROOT/models/siglip2-base-patch16-256
QVV=$AUDIT/rscc_ebd/rscc_ebd_qvq_caption_pilot.jsonl
HUMAN=$ROOT/manifests/qcpr_dataset_v2_expanded_benchmark_157d292_20260728/registries/caption_registry.jsonl
SHA=$(git -C "$WORKTREE" rev-parse HEAD)
RUN_ROOT=$AUDIT/rscc_ebd/siglip2_verification_pilot_$SHA
mkdir -p "$RUN_ROOT/logs"
test -z "$(git -C "$WORKTREE" status --porcelain)"
JID=$(sbatch --parsable --partition=research --job-name=qcpr-s2-qvq-verify --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G --time=02:00:00 --output="$RUN_ROOT/logs/slurm-%j.out" --wrap="set -eu; cd $WORKTREE; test \$(git rev-parse HEAD) = $SHA; test -z \"\$(git status --porcelain)\"; export PYTHONPATH=$WORKTREE/src:$WORKTREE/scripts; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec $PYTHON scripts/verify_rscc_qvq_siglip2_pilot.py --model-path $MODEL --qvq-pilot $QVV --human-caption-registry $HUMAN --output-dir $RUN_ROOT --pilot 500 --calibration 100 --batch-size 8")
printf "RUN_ROOT=%s\nJOB_ID=%s\nSHA=%s\n" "$RUN_ROOT" "$JID" "$SHA"
squeue -j "$JID" -o "%.18i %.9T %.30j %.10M %R"
