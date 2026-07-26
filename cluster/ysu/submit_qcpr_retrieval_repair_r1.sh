#!/usr/bin/env bash
set -euo pipefail

PROJECT=/mnt/weka/svardanyan/rs_change_project
WORKTREE=$PROJECT/code/project-qcpr-retrieval-repair-r1
MANIFEST_DIR=$PROJECT/manifests/qcpr_v3_clean_058779b7
PYTHON=$PROJECT/envs/rschange/bin/python
BASELINE=$PROJECT/runs/qcpr_residual_pair_grounding_20260726-155559/retrieval/best_retrieval.pt
REPRODUCTION_ROOT=$PROJECT/runs/qcpr_retrieval_repair_r0_20260727/reproduction
AUDIT_ROOT=$PROJECT/runs/qcpr_retrieval_repair_r0_20260727/audit
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_ROOT=${RUN_ROOT:-$PROJECT/runs/qcpr_retrieval_repair_r1_$STAMP}
SHA=$(git -C "$WORKTREE" rev-parse HEAD)
BRANCH=$(git -C "$WORKTREE" branch --show-current)

test "$BRANCH" = codex/qcpr-retrieval-repair-r1
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -f "$REPRODUCTION_ROOT/reproduction.json"
test -f "$AUDIT_ROOT/retrieval_identifiability_audit.json"
test "$($PYTHON -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$REPRODUCTION_ROOT/reproduction.json")" = true

TRAIN_HASH=$(sha256sum "$MANIFEST_DIR/natural_train_retrieval_manifest.jsonl" | awk '{print $1}')
VAL_HASH=$(sha256sum "$MANIFEST_DIR/natural_validation_retrieval_manifest.jsonl" | awk '{print $1}')
AUDIT_HASH=$(sha256sum "$MANIFEST_DIR/caption_quality_audit.jsonl" | awk '{print $1}')
BASELINE_HASH=$(sha256sum "$BASELINE" | awk '{print $1}')
REPRODUCTION_HASH=$(sha256sum "$REPRODUCTION_ROOT/reproduction.json" | awk '{print $1}')
IDENTIFIABILITY_HASH=$(sha256sum "$AUDIT_ROOT/retrieval_identifiability_audit.json" | awk '{print $1}')

mkdir -p "$RUN_ROOT"/{r1,logs,hard_negatives,report}
"$PYTHON" - "$RUN_ROOT" "$SHA" "$BRANCH" "$TRAIN_HASH" "$VAL_HASH" "$AUDIT_HASH" "$BASELINE_HASH" "$REPRODUCTION_HASH" "$IDENTIFIABILITY_HASH" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
root.joinpath("run_contract.json").write_text(json.dumps({
 "stage":"R1", "git_sha":sys.argv[2], "branch":sys.argv[3],
 "manifest_sha256":{"train":sys.argv[4],"validation":sys.argv[5],"caption_audit":sys.argv[6]},
 "baseline_checkpoint_sha256":sys.argv[7], "reproduction_sha256":sys.argv[8],
 "identifiability_audit_sha256":sys.argv[9], "fresh_optimizer":True,
 "partial_backbone_adaptation":False, "local_token_patch_loss_weight":0.0,
 "physical_micro_batch":16, "logical_physical_batch":128,
 "captions_per_pair":2, "logical_queries":256, "logical_score_matrix":[256,128],
 "gradcache_recomputation":True, "gradient_accumulation_is_not_contrastive_batch":True,
 "hard_negative_warmup_epochs":3, "hard_negative_refresh_epochs":2,
 "hard_negative_cache_size":32, "masks_during_training":False,
 "automatic_promotion_to_r2":False
},indent=2,sort_keys=True)+"\n")
PY

COMMON="set -euo pipefail; cd $WORKTREE; test \$(git rev-parse HEAD) = $SHA; test -z \"\$(git status --porcelain)\"; test \$(sha256sum $BASELINE | cut -c1-64) = $BASELINE_HASH; test \$(sha256sum $MANIFEST_DIR/natural_train_retrieval_manifest.jsonl | cut -c1-64) = $TRAIN_HASH; test \$(sha256sum $MANIFEST_DIR/natural_validation_retrieval_manifest.jsonl | cut -c1-64) = $VAL_HASH; export PYTHONPATH=$WORKTREE/src:$WORKTREE/scripts; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
JOB_ID=$(sbatch --parsable --partition=research --job-name=qcpr-repair-r1 --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=48:00:00 --output="$RUN_ROOT/logs/r1-%j.out" --wrap="$COMMON; $PYTHON scripts/train_qcpr_retrieval_repair_r1.py --baseline-checkpoint $BASELINE --reproduction $REPRODUCTION_ROOT/reproduction.json --identifiability-audit $AUDIT_ROOT/retrieval_identifiability_audit.json --output-dir $RUN_ROOT/r1 --manifest-dir $MANIFEST_DIR --physical-micro-batch 16 --logical-physical-batch 128 --captions-per-pair 2 --hard-candidates 32 --hard-warmup-epochs 3 --hard-refresh-epochs 2 --epochs 12 --min-epochs 4 --patience 3 --local-loss-weight 0")

printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'R1_JOB_ID=%s\n' "$JOB_ID"
printf 'GIT_SHA=%s\n' "$SHA"
squeue -j "$JOB_ID" -o "%.18i %.9T %.30j %.10M %.2t %R"
