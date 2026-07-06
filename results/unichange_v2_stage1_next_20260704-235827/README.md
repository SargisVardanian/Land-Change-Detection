# UniChange v2 Stage-1-next mixed semantic run

## Pipeline

- Git commit: `cc8549bacbf322a32b3306e88255caace3b8f007`
- Smoke job: `86155`
- Memory job: `86156`
- Training job: `86157`
- Evaluation job: `86158`

## Dataset configuration

- LEVIR-MCI sampling weight: `0.55`
- SECOND-CC sampling weight: `0.45`
- Temporal depth: `6`
- Text maximum length: `256`
- Physical batch size: `32`

## Slurm results

- Smoke: `COMPLETED 0:0`
- Memory probe: `COMPLETED 0:0`
- Training: `COMPLETED 0:0`
- Evaluation: `COMPLETED 0:0`

## Runtime directories

- Smoke: `/mnt/weka/svardanyan/rs_change_project/runs/unichange_v2_semantic_smoke_20260704-235827`
- Memory: `/mnt/weka/svardanyan/rs_change_project/runs/unichange_v2_memory_probe_20260704-235827`
- Training: `/mnt/weka/svardanyan/rs_change_project/runs/unichange_v2_semantic_train_20260704-235827`
- Evaluation: `/mnt/weka/svardanyan/rs_change_project/runs/unichange_v2_semantic_train_20260704-235827/evaluation`

## Excluded artifacts

The model checkpoints are intentionally not committed to Git:

- `best_retrieval.pt`
- `last_retrieval.pt`

Each checkpoint is approximately 2.35 GB and exceeds normal GitHub file limits.
