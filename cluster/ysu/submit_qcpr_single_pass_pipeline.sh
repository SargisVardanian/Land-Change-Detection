#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/weka/svardanyan/rs_change_project
WORKTREE=$PROJECT/code/project-qcpr-single-pass-cls-localization
MANIFEST_DIR=$PROJECT/manifests/qcpr_v3_clean_058779b7
PYTHON=$PROJECT/envs/rschange/bin/python
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_ROOT=${RUN_ROOT:-$PROJECT/runs/qcpr_residual_pair_grounding_$STAMP}
SHA=$(git -C "$WORKTREE" rev-parse HEAD)
BRANCH=$(git -C "$WORKTREE" branch --show-current)
REMOTE_PUSH_STATUS=${REMOTE_PUSH_STATUS:-not_attempted}
test -z "$(git -C "$WORKTREE" status --porcelain)"
TRAIN_HASH=$(sha256sum "$MANIFEST_DIR/natural_train_retrieval_manifest.jsonl" | awk '{print $1}')
VAL_HASH=$(sha256sum "$MANIFEST_DIR/natural_validation_retrieval_manifest.jsonl" | awk '{print $1}')
AUDIT_HASH=$(sha256sum "$MANIFEST_DIR/caption_quality_audit.jsonl" | awk '{print $1}')
mkdir -p "$RUN_ROOT"/{retrieval,grounding,evaluation,report,logs,examples}
"$PYTHON" - "$RUN_ROOT" "$SHA" "$BRANCH" "$REMOTE_PUSH_STATUS" "$TRAIN_HASH" "$VAL_HASH" "$AUDIT_HASH" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
(root/"run_contract.json").write_text(json.dumps({
 "architecture":"joint_universat_residual_pair_adapter_plus_mask_free_grounding",
 "git_sha":sys.argv[2],"branch":sys.argv[3],"remote_push_status":sys.argv[4],
 "manifest_sha256":{"train":sys.argv[5],"validation":sys.argv[6],"caption_audit":sys.argv[7]},
 "output_grid":32,"masks_during_training":False,
 "initial_retrieval_batch":{"physical_items":16,"captions_per_item":2,"queries":32,"accumulation":4,"effective_physical_items":64},
 "retrieval_oom_fallback":{"physical_items":8,"captions_per_item":2,"queries":16,"accumulation":8,"effective_physical_items":64},
 "backbone_audit":{"input_shape":[1,2,3,256,256],"joint_output_shape":[1,1024,768],"finite":True,"api_supports_arbitrary_T":True},
 "chain":["qcpr-retrieval-full","qcpr-query-grounding-full","qcpr-evaluation","qcpr-report"]},indent=2)+"\n")
PY
COMMON="set -eu; cd $WORKTREE; test \$(git rev-parse HEAD) = $SHA; test -z \$(git status --porcelain); test \$(sha256sum $MANIFEST_DIR/natural_train_retrieval_manifest.jsonl | cut -c1-64) = $TRAIN_HASH; test \$(sha256sum $MANIFEST_DIR/natural_validation_retrieval_manifest.jsonl | cut -c1-64) = $VAL_HASH; test \$(sha256sum $MANIFEST_DIR/caption_quality_audit.jsonl | cut -c1-64) = $AUDIT_HASH; export PYTHONPATH=$WORKTREE/src:$WORKTREE/scripts; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
J1=$(sbatch --parsable --partition=research --job-name=qcpr-retrieval-full --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=48:00:00 --output="$RUN_ROOT/logs/retrieval-%j.out" --wrap="$COMMON; $PYTHON scripts/train_qcpr_single_pass.py --output-dir $RUN_ROOT/retrieval --manifest-dir $MANIFEST_DIR --output-grid 32 --batch-size 16 --accumulation 4")
J2=$(sbatch --parsable --partition=research --dependency=afterok:$J1 --job-name=qcpr-query-grounding-full --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=48:00:00 --output="$RUN_ROOT/logs/grounding-%j.out" --wrap="$COMMON; test -f $RUN_ROOT/retrieval/best_retrieval.pt; test -f $RUN_ROOT/retrieval/retrieval_acceptance.json; $PYTHON scripts/train_qcpr_query_localization.py --retrieval-checkpoint $RUN_ROOT/retrieval/best_retrieval.pt --retrieval-acceptance $RUN_ROOT/retrieval/retrieval_acceptance.json --output-dir $RUN_ROOT/grounding --manifest-dir $MANIFEST_DIR --output-grid 32 --query-batch 8 --candidates 4 --hard-cache-size 32 --validation-candidates 16")
J3=$(sbatch --parsable --partition=research --dependency=afterok:$J2 --job-name=qcpr-evaluation --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=24:00:00 --output="$RUN_ROOT/logs/evaluation-%j.out" --wrap="$COMMON; test -f $RUN_ROOT/grounding/best_grounding.pt; $PYTHON scripts/evaluate_qcpr_single_pass.py --grounding-checkpoint $RUN_ROOT/grounding/best_grounding.pt --output-dir $RUN_ROOT/evaluation --manifest-dir $MANIFEST_DIR --output-grid 32")
J4=$(sbatch --parsable --partition=research --dependency=afterok:$J3 --job-name=qcpr-report --cpus-per-task=2 --mem=16G --time=02:00:00 --output="$RUN_ROOT/logs/report-%j.out" --wrap="$COMMON; test -f $RUN_ROOT/evaluation/evaluation.json; $PYTHON scripts/report_qcpr_single_pass.py --run-root $RUN_ROOT")
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'RETRIEVAL_JOB_ID=%s\n' "$J1"
printf 'LOCALIZATION_JOB_ID=%s\n' "$J2"
printf 'EVALUATION_JOB_ID=%s\n' "$J3"
printf 'REPORT_JOB_ID=%s\n' "$J4"
printf 'DEPENDENCY_CHAIN=%s->%s->%s->%s\n' "$J1" "$J2" "$J3" "$J4"
squeue -j "$J1,$J2,$J3,$J4" -o "%.18i %.9T %.30j %.10M %.2t %R"
