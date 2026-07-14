# QCPR v3 runtime tensor-device contract

The data collator deliberately returns Python metadata (`captions`, image paths,
pair IDs and dataset names) on CPU.  It may also return tensor metadata on CPU.
Before model forward, a loss calculation or any tensor indexing operation, every
tensor participating in arithmetic or indexing is moved to the selected
`torch.device`.

`retrieval_supervision_selection` is the canonical boundary for retrieval
filtering.  It moves `retrieval_supervision` and `caption_to_pair` to the
requested device before indexing, validates both as rank-1, validates mapping
range, constructs compact selected indices with `index_select`/`index_copy_`,
rejects negative compact mappings, and verifies every returned tensor device.

Audit scope for RC1 repair:

| Path | Boundary result |
|---|---|
| `scripts/train_qcpr_v3.py` | images, mapping and temporal mask move to CUDA; selection now moves collator metadata internally; captions remain Python metadata. |
| `scripts/ucv2_stage1_next_core.py` | canonical retrieval selection repaired; downstream tensors index model outputs on the same device. |
| `scripts/ucv2_stage1_next_smoke_core.py` | reuses canonical selection; captions are converted to Python only after `selected_queries.tolist()`. |
| `scripts/ucv2_retrieval_metrics.py` | corpus tensors are explicitly accumulated on CPU after evaluation; CPU indexing is therefore intentional. |
| `src/land_change_detection/models/qcpr_v3*` | forward/scoring inputs are tensor-only and expected to share the selected device; no CPU metadata tensor is used for model indexing. |

The H100 smoke asserts the CUDA device for every retrieval-selection output,
including a CPU-collated batch.  Regression tests cover CPU-only behavior and
run the same case on CUDA when available.
