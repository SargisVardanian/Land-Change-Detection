# QCPR v3 cleanup manifest

This manifest classifies the QCPR surface after the clean-bootstrap correction.
Obsolete v2 behavior is preserved only for historical compatibility. V3 model
construction no longer instantiates `UniChangeV2RetrievalModel`; temporary
reuse of legacy dataset/collator utilities is isolated as a data compatibility
boundary and does not define the v3 architecture.

## Model and backbone components

| Path/component | Classification | RC1 action |
|---|---|---|
| `models/temporal_change_encoder.py` | migrate into v3 | preserve v1 semantics; expose multi-scale v3 inputs separately |
| `models/retrieval_heads.py` global projection/loss | keep unchanged | reused by v1-compatible global path |
| `models/retrieval_heads.py` caption parser | evaluation-only | never passed to v3 neural model |
| `models/unichange_v2_retrieval.py` | v1/v2 compatibility only | historical loading/evaluation only; not used by clean v3 factory |
| `models/qcpr.py::score` | v1 compatibility only | retained for old checkpoints |
| `models/qcpr.py::score_v2` and semantic-specific scores | obsolete for v3 | no v3 import; replaced by generic v3 model/API |
| `backbones/jina_v5_text.py` global/token encoder | keep unchanged | metadata semantic token groups ignored by v3 model |
| `models/qcpr_v3.py` | migrate into v3 | new generic field/decoder/mask/slots/canonical scoring |
| `models/qcpr_v3_teacher.py` | historical compatibility | strict historical teacher only when an explicit checkpoint exists; inactive for clean bootstrap |
| `models/qcpr_v3_factory.py` | v3 primary | directly builds Jina, UniverSat, temporal encoder, global head and generic grounder |

## Training/evaluation scripts

| Paths | Classification | RC1 action |
|---|---|---|
| `train_unichange_v2_retrieval*.py`, model-building `ucv2_*` | v1/v2 compatibility only | retained for history; clean v3 factory does not call them |
| legacy temporal-caption dataset/collator utilities | temporary data compatibility | may be called by v3 entrypoints only for manifest IO/collation until moved to a neutral data module |
| `render_unichange_v2_retrieval.py` | v1/v2 result readability | retained; v3 renderer is separate and canonical-API based |
| `compare_qcpr_evaluations.py`, plotting/summarizers | evaluation-only | retained |
| manifest builders and temporal-caption audit scripts | keep unchanged | reused as immutable source adapters |
| `qcpr_v3_core.py` | migrate into v3 | new validated phases, training, validation and artifact contract |
| `train_qcpr_v3.py`, `evaluate_qcpr_v3.py`, `render_qcpr_v3.py` | migrate into v3 | sole v3 entrypoints |
| `build_qcpr_v3_dataset.py` | migrate into v3 | derived manifests/audits only |
| `build_qcpr_v3_benchmark.py` | migrate into v3 | deterministic unreviewed benchmark package |

## Commands and cluster wrappers

| Paths | Classification | RC1 action |
|---|---|---|
| `commands/stage1_next_v2_patchseg_*` | generated/obsolete wrappers | remove; exact history remains in git and archived runs |
| legacy `cluster/ysu/*unichange_v2*` | v1/v2 compatibility only | retain for reproducibility, clearly not v3 entrypoints |
| `cluster/ysu/qcpr_v3_smoke.sbatch` | migrate into v3 | one comprehensive post-push smoke |
| `cluster/ysu/qcpr_v3_memory.sbatch` | migrate into v3 | full-step, three-step post-smoke probe |
| `cluster/ysu/qcpr_v3_experiment.sbatch` | migrate into v3 | gated B/C diagnostics/pilot only |

## Tests

| Tests | Classification | RC1 action |
|---|---|---|
| v1 retrieval/checkpoint/renderer tests | keep unchanged | protect immutable baseline readability |
| v2 semantic-head tests | v1/v2 compatibility only | retained unless directly asserting v3 behavior |
| source-string tests in `test_ucv2_stage1_next_training_core.py` | obsolete duplicate | replace affected v3 requirements with functional phase/API tests; retain legacy test until replacement passes, then remove only source-inspection cases |
| manifest, dataset, retrieval metric tests | keep unchanged | regression coverage |
| `test_qcpr_v3_*.py` | migrate into v3 | functional model, phase, teacher, dataset, benchmark, parity and loss tests |

## Documentation and artifacts

| Paths | Classification | RC1 action |
|---|---|---|
| `docs/qcpr_v2_*`, `UNICHANGE_V2_*` | v1/v2 historical compatibility | retain and label historical by context |
| v3 scientific/design documents | migrate into v3 | finalize in RC1 |
| files under `runs/`, canonical manifests/raw data | immutable external artifacts | never modify/delete |
| generated caches, plots, checkpoints inside worktree | generated artifact | reject/remove before commit; none present at RC1 start |

## Explicit removals planned in RC1

The six `commands/stage1_next_v2_patchseg_*` wrappers are duplicate convenience
launchers tied to the retired v2 experiment. Their underlying historical
scripts and Slurm files remain for reproducibility. No test is deleted merely
because it is old. Source-string assertions are removed only when the new
functional phase/entrypoint tests cover the same safety contract.
