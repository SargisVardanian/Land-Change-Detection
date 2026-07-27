# QCPR v3 RC1 reproducibility baseline

Recorded before implementation on 2026-07-14.

| Field | Value |
|---|---|
| Host | `cluster` |
| User | `svardanyan` |
| Source worktree | `/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-98e-cluster` |
| RC1 worktree | `/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1` |
| Source branch | `codex/qcpr-v3-generic-grounding` |
| RC1 branch | `codex/qcpr-v3-rc1-experiment` |
| Starting SHA | `d54072fd7206b185e773677a194d9cbb81157ff6` |
| Source remote SHA | absent: source research branch was not pushed |
| RC1 remote SHA | absent before RC1 commit/push |
| Python | `/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python` |
| Python version | `3.11.15` |
| PyTorch | `2.12.1+cu130` |
| PyTorch CUDA build | `13.0` |
| Login-node CUDA available | `false` (expected; GPU validation uses Slurm only after tests/push) |

The source HEAD had no tracked modifications. It had three untracked v3 design
documents, which were intentionally copied into the isolated RC1 worktree. No
dataset, manifest, checkpoint, run, or archived evaluation artifact was copied,
modified, or deleted.

Immutable teacher/initialization checkpoint:

`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_e0_20260711-015629/pilot/best_retrieval.pt`

Archived v2 diagnostic `99570` remains read-only and is not an initialization
source for v3.
