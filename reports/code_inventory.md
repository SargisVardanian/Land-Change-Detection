# Dataset code inventory

## Baseline

The repository contains 681 non-cache files and approximately 89,983 lines: 523 Python
files (77,771 lines), 61 Slurm files, 44 Markdown files and 30 shell files. Top-level
concentrations are `scripts/` 219 files/40,176 lines, `src/` 180/22,788,
`tests/` 125/13,614, `cluster/` 87/3,118 and `docs/` 43/6,659.

## Classification

| Class | Scope | Disposition |
|---|---|---|
| ACTIVE_CANONICAL | `src/qcpr_data/contracts`, `identities`, `manifests`, `queries`, `reports`, `sources`, `splits`, `verification` | maintained reusable Dataset-v2 library |
| ACTIVE_TOOL | `scripts/validate_qcpr_release_contract.py` and source acquisition/audit tools that import canonical modules | supported operational entrypoints |
| ACTIVE_TEST | `tests/test_qcpr_data_contracts.py`, Dataset-v2/retrieval/source tests and validator safety test | retained verification |
| LEGACY_REQUIRED | immutable release builders/finalizers and historical audit scripts | retained for lineage; not a production entrypoint |
| GENERATED | `__pycache__`, `.pyc`, pytest XML/cache | remove when untracked; regenerate outside worktree |
| OBSOLETE | temporary repair/scratch outputs proven untracked and unreferenced | none tracked in this audit |

## Canonical ownership

| Concern | Canonical implementation |
|---|---|
| schemas/validation | `src/qcpr_data/contracts/` |
| physical/frame hashing and overlap | `src/qcpr_data/identities/` |
| release layout and checksum verification | `src/qcpr_data/manifests/` |
| normalization, purpose, attributes, collisions | `src/qcpr_data/queries/purpose.py` |
| exact/semantic/localized/direction/stable/series views | corresponding `src/qcpr_data/queries/*.py` |
| split policy | `src/qcpr_data/splits/` |
| reviewer/adjudication promotion | `src/qcpr_data/verification/` |
| final immutable release validation | `scripts/validate_qcpr_release_contract.py` (read-only by default) |

Historical `build_qcpr_final_r19.py`, `finalize_qcpr_bitemporal_final.py`,
`finalize_qcpr_dataset_v2.py` and retrieval-repair packagers are classified
LEGACY_REQUIRED because immutable artifacts cite them or their behavior. They contain
release-specific orchestration and must not be imported as the general library.

No r18/r19 or workstation-specific paths were found in `src/qcpr_data`. Exact release
paths remain intentionally present in immutable contract snapshots and historical
reports. Model-stage scripts are outside this dataset cleanup scope.
