# YSU-HPC Actual Runbook

Run everything below inside the YSU Portal `Interactive -> Shell Terminal with Conda` session.

## Project Root

```bash
if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
    export RS_PROJECT_ROOT="/mnt/weka/$USER/rs_change_project"
else
    export RS_PROJECT_ROOT="/data/$USER/rs_change_project"
fi
```

## Clone Or Refresh The Branch

```bash
mkdir -p "$RS_PROJECT_ROOT/code"
cd "$RS_PROJECT_ROOT/code"

git clone -b codex/ysu-hpc-bootstrap-and-verification \
  https://github.com/SargisVardanian/Land-Change-Detection.git project \
  || true

cd project
git fetch origin
git checkout codex/ysu-hpc-bootstrap-and-verification
git pull origin codex/ysu-hpc-bootstrap-and-verification
```

## Environment And Dataset Bootstrap

```bash
bash cluster/ysu/setup_rschange_env.sh
bash cluster/ysu/download_change_retrieval_datasets.sh
bash cluster/ysu/bootstrap_change_retrieval_assets.sh
bash cluster/ysu/verify_project_assets.sh
```

## First Real Experiment Jobs

```bash
sbatch cluster/ysu/smoke_report.sbatch
sbatch cluster/ysu/validate_levir_mci_dataset.sbatch
sbatch cluster/ysu/render_levir_mci_samples.sbatch
sbatch cluster/ysu/overfit_levir_mci_100.sbatch
```

## Checks While Jobs Run

```bash
squeue -u $USER
tail -f logs/*.out
du -sh "$RS_PROJECT_ROOT/datasets/raw/"*
df -h /mnt/weka
df -h /data
```

## Success Criteria

1. LEVIR-MCI real files parse successfully.
2. `runs/levir_mci_grid.png` is generated from real data.
3. `runs/levir_mci_overfit/` exists and contains a checkpoint plus metrics history.
4. Overfit-100 reaches train Dice above `0.90`.
5. Full training reports Dice, IoU, Precision, Recall, Recall@1/5/10, and MRR.
6. Retrieval `Recall@5` is clearly above random.
7. Predicted masks visually match real ground-truth masks.

Do not call the project complete until these cluster artifacts exist:

- `$RS_PROJECT_ROOT/runs/levir_mci_grid.png`
- `$RS_PROJECT_ROOT/runs/levir_mci_overfit/`
- `$RS_PROJECT_ROOT/logs/`
- `$RS_PROJECT_ROOT/indexes/`
- `$RS_PROJECT_ROOT/datasets/dataset_manifest.md`
- `$RS_PROJECT_ROOT/rs_change_project_verification.json`
- `$RS_PROJECT_ROOT/runs/cluster_smoke_report.json`
