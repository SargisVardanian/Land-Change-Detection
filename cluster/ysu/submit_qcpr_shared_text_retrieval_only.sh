#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project
WORKTREE=$ROOT/code/project-qcpr-shared-text-retrieval-only
MANIFEST_DIR=$ROOT/manifests/qcpr_v3_clean_058779b7
S2_DIR=$ROOT/manifests/qcpr_v31_m0_a936f116
SHA=$(git -C "$WORKTREE" rev-parse HEAD)
TRAIN_HASH=$(sha256sum "$MANIFEST_DIR/natural_train_retrieval_manifest.jsonl" | awk '{print $1}')
VAL_HASH=$(sha256sum "$MANIFEST_DIR/natural_validation_retrieval_manifest.jsonl" | awk '{print $1}')
SOFTMAP_HASH=$(sha256sum "$S2_DIR/m0_dev_manifest.jsonl" | awk '{print $1}')
RUN=$ROOT/runs/qcpr_shared_text_retrieval_only_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"/{a0,b,evaluation,report}
cat > "$RUN/run_contract.json" <<EOF
{"git_sha":"$SHA","parent_sha":"931be9220f54b429ce9001503178b4be65f85452","worktree":"$WORKTREE","visual_architecture":"unchanged_from_parent","text_adapter":"SharedTextAttentionAdapter","training":"A0_global_plus_B_token_patch_only","segmentation":"evaluation_only_emergent_patch_evidence","physical_batch":32,"manifest_dir":"$MANIFEST_DIR","train_manifest_sha256":"$TRAIN_HASH","validation_manifest_sha256":"$VAL_HASH","softmap_manifest_sha256":"$SOFTMAP_HASH"}
EOF
git -C "$WORKTREE" remote -v > "$RUN/git_remote.txt"
env | sort > "$RUN/environment.txt"
common="source \$HOME/rschange_env.sh; cd $WORKTREE; export PYTHONPATH=$WORKTREE/src:$WORKTREE/scripts"
J1=$(sbatch --parsable --job-name=qcpr-text-a0 --gres=gpu:1 --cpus-per-task=12 --mem=100G --time=48:00:00 --wrap="$common; /mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python scripts/train_qcpr_emergent_epochs.py --track A0 --output-dir $RUN/a0 --manifest-dir $MANIFEST_DIR --max-epochs 30 --min-epochs 10 --patience 5 --batch-size 32 --learning-rate 5e-5 --text-adapter-learning-rate 5e-6 --soft-map-manifest $S2_DIR/m0_dev_manifest.jsonl --soft-map-queries 16")
J2=$(sbatch --parsable --dependency=afterok:$J1 --job-name=qcpr-text-b --gres=gpu:1 --cpus-per-task=12 --mem=100G --time=48:00:00 --wrap="$common; test -f $RUN/a0/best_feasible.pt; /mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python scripts/train_qcpr_emergent_epochs.py --track B --output-dir $RUN/b --manifest-dir $MANIFEST_DIR --initialization-checkpoint $RUN/a0/best_feasible.pt --baseline-report $RUN/a0/best_feasible_development.json --max-epochs 30 --min-epochs 10 --patience 6 --batch-size 32 --soft-map-manifest $S2_DIR/m0_dev_manifest.jsonl --soft-map-queries 16")
J3=$(sbatch --parsable --dependency=afterok:$J2 --job-name=qcpr-text-eval --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=12:00:00 --wrap="$common; test -f $RUN/b/best_feasible.pt; /mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python scripts/evaluate_qcpr_c0_emergent.py --checkpoint $RUN/b/best_feasible.pt --manifest $S2_DIR/m0_dev_manifest.jsonl --output-dir $RUN/evaluation")
J4=$(sbatch --parsable --dependency=afterok:$J3 --job-name=qcpr-text-report --cpus-per-task=2 --mem=8G --time=01:00:00 --wrap="$common; /mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python scripts/build_qcpr_shared_text_report.py --run-root $RUN")
printf '%s\n' "RUN=$RUN" "SHA=$SHA" "A0=$J1" "B=$J2" "EVAL_ONLY=$J3" "REPORT=$J4"
