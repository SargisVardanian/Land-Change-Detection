# Research Execution Board

## Architectural Goal

Build an additive research subsystem for retrieval and research-pipeline orchestration without changing the current normal-mode Streamlit demo contract.

Primary rule:

`existing demo contract stays fixed` + `new research-layer grows beside it`

Research direction:

`segmentation evidence -> retrieval evidence -> structured evidence bundle -> optional explainer`

## Retrieval Modes

- `static_region`
- `pair_analog`
- `text_bitemporal`
- `novelty_single_image`
- `transition_conditioned`
- `trajectory`

## New Packages And Files

### Phase 1

- `src/land_change_detection/retrieval/`
  - `__init__.py`
  - `contracts.py`
  - `runtime.py`
  - `registry.py`
  - `evidence.py`
  - `backends/__init__.py`
  - `backends/base.py`
  - `backends/fake.py`
  - `backends/noop.py`
  - `indexing/__init__.py`
  - `indexing/manifests.py`
  - `indexing/vector_store.py`
- `tests/test_retrieval_core.py`

### Phase 2

- `src/land_change_detection/pipelines/research_pipeline.py`
- `src/land_change_detection/pipelines/evidence_bundle.py`
- `tests/test_research_pipeline.py`

### Planned Later Phases

- `pages/Research_Retrieval.py`
- `src/land_change_detection/ingestion/`
- `src/land_change_detection/training/`
- retrieval backends for static, pair, and text-conditioned retrieval
- experimental Prithvi path
- benchmark and dataset documentation

## Ownership

### subagent_retrieval_core

Owns only:

- `src/land_change_detection/retrieval/**`
- `tests/test_retrieval_core.py`

Must not edit:

- `app.py`
- `src/land_change_detection/pipeline_v2.py`
- existing segmentation backend files
- existing Streamlit demo code

### subagent_research_pipeline

Owns only:

- `src/land_change_detection/pipelines/research_pipeline.py`
- `src/land_change_detection/pipelines/evidence_bundle.py`
- `tests/test_research_pipeline.py`

May read but should avoid changing:

- `src/land_change_detection/contracts.py`
- `src/land_change_detection/research_framework.py`
- `src/land_change_detection/pipeline_v2.py`

Must not edit:

- `app.py`
- retrieval core ownership files

## Order Of Work

1. Create architecture board and freeze constraints.
2. Implement retrieval core contracts, registry, runtime, and deterministic fake/no-op backends.
3. Implement research pipeline and evidence bundle using fake retrieval runtime.
4. Run targeted tests for new modules.
5. Run existing regression tests that protect the normal app contract.
6. Only after Phase 1 and Phase 2 are green, proceed to UI, ingestion, and real backends.

## Acceptance Criteria

### Global

- `app.py` normal mode remains behaviorally unchanged.
- `tests/test_streamlit_vlm_first_contract.py` stays green.
- heavy dependencies are optional and guarded.
- no real model weights are required for Phase 1 or Phase 2 tests.
- all new dataclasses and evidence structures are deterministic and serializable.

### Phase 1

- retrieval backend registration works
- retrieval runtime dispatch works
- fake backend returns deterministic top-k
- no-op backend returns empty but valid retrieval result
- vector store and manifest helpers import without heavy dependencies

### Phase 2

- research pipeline composes segmentation artifacts and retrieval artifacts
- evidence bundle remains structured and deterministic
- LLM serializer receives only controlled fields
- no raw prompt blobs or model-private internals are passed through

## Protected Areas

These parts are contract-frozen and should not be modified during Phase 1 or Phase 2 unless a test-guided necessity appears:

- [app.py](/Users/sargisvardanyan/Land-Change-Detection/app.py)
- [tests/test_streamlit_vlm_first_contract.py](/Users/sargisvardanyan/Land-Change-Detection/tests/test_streamlit_vlm_first_contract.py)
- [src/land_change_detection/pipeline_v2.py](/Users/sargisvardanyan/Land-Change-Detection/src/land_change_detection/pipeline_v2.py)
- [src/land_change_detection/segmentation_backends/prithvi_terratorch.py](/Users/sargisvardanyan/Land-Change-Detection/src/land_change_detection/segmentation_backends/prithvi_terratorch.py)

## Tests That Must Stay Green

- `tests/test_streamlit_vlm_first_contract.py`
- `tests/test_research_framework.py`
- `tests/test_contracts.py`
- `tests/test_pipeline_v2_smoke.py`
- `tests/test_prithvi_backend_placeholder.py`

## Test Command

Primary local command:

```bash
PYTHONPATH=src python -m pytest -q
```

For Phase 1 and Phase 2 iteration, run targeted subsets first and then the regression subset above.

## Non-Goals For Phase 1 And Phase 2

- no real Prithvi inference
- no RemoteCLIP dependency requirement
- no FAISS requirement
- no Sentinel credential flow
- no dataset downloads
- no modification of user-facing normal demo explanation flow
