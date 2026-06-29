#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$CODE_ROOT"

RUN_DIR="$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100"
mkdir -p "$CODE_ROOT/logs" "$RUN_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "Python not executable: $PYTHON" >&2
  exit 2
fi

export MCI_ROOT="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export USE_BF16="${USE_BF16:-1}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

jina_job=$(sbatch --parsable "$SCRIPT_DIR/probe_jina_v5.sbatch")
universat_job=$(sbatch --parsable "$SCRIPT_DIR/probe_universat.sbatch")
smoke_job=$(sbatch --parsable --dependency=afterok:${jina_job}:${universat_job} "$SCRIPT_DIR/smoke_unichange_joint.sbatch")
overfit_job=$(sbatch --parsable --dependency=afterok:${smoke_job} "$SCRIPT_DIR/overfit_unichange_mci_100.sbatch")
eval_job=$(sbatch --parsable --dependency=afterok:${overfit_job} "$SCRIPT_DIR/eval_unichange_mci_100.sbatch")
render_job=$(sbatch --parsable --dependency=afterok:${eval_job} "$SCRIPT_DIR/render_unichange_mci_100.sbatch")
plot_job=$(sbatch --parsable --dependency=afterok:${eval_job} --wrap="cd '$CODE_ROOT' && PYTHONPATH='$CODE_ROOT/src:$CODE_ROOT' '$PYTHON' '$CODE_ROOT/scripts/plot_unichange_history.py' --run-dir '$RUN_DIR'")

cat > "$RUN_DIR/pipeline_jobs.env" <<EOF
JINA_JOB=$jina_job
UNIVERSAT_JOB=$universat_job
SMOKE_JOB=$smoke_job
OVERFIT_JOB=$overfit_job
EVAL_JOB=$eval_job
RENDER_JOB=$render_job
PLOT_JOB=$plot_job
EOF

cat "$RUN_DIR/pipeline_jobs.env"
