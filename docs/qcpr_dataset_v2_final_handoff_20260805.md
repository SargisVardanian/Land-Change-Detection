# QCPR Dataset-v2 final handoff — 2026-08-05

Decision: **DATA_QUALITY_HOLD**

## Provenance

- Branch: `codex/qcpr-dataset-v2-final`
- Branch HEAD at generation: `dc33c89aa801e8c7f125290fbe3d80719291f743`
- Immutable release code SHA: `97bee86b3e137460c419078c76bd6cd46b94c768`
- Release: `/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_dataset_v2_final_97bee86_20260804`
- Release SHA256SUMS digest: `eb3a2e14b5467eec4da3323b06bb71de43a8a2576ca8ded380806b7503951442` (60 entries)
- Worktree clean at generation: `True`
- Disk: `73 TiB free / 272 TiB total`

## Physical and query counts

- Physical items: `32548`; frames: `66074`; sequences: `489`
- Items by source: `{'levir_mci': 8143, 'rscc_ebd': 18215, 's2looking': 1000, 'second_cc': 4701, 'tamms': 489}`
- Queries: `66452`; training-enabled: `61080`
- Query scopes: `{'direction': 19015, 'exact': 43839, 'localized': 2000, 'semantic': 1598}`
- Verification: `{'derived_eval': 3598, 'human': 62854}`

## View counts

| View | train | development | test | status |
|---|---:|---:|---:|---|
| exact | 37529 | 5065 | 1245 | READY_EXACT_ONLY_CONDITIONAL_SOURCE_TERMS |
| semantic | 0 | 0 | 1598 | HOLD |
| localized | 0 | 0 | 2000 | EVAL_ONLY |
| direction | 16352 | 2184 | 479 | EXACT_CONDITIONAL |
| stable | 0 | 0 | 0 | EMPTY_GENERIC_NO_CHANGE_EXCLUDED |
| long_series | 0 | 0 | 0 | PILOT_PHYSICAL_ONLY_TEXT_HOLD |

## Official sources and licenses

| Source | State | License status | Training |
|---|---|---|---|
| ChangeChat-87k | ACCESS_REQUIRED | REVIEW_REQUIRED | False |
| DUBAI-CC | OFFICIAL_LINK_FOUND_ARCHIVE_NOT_ACQUIRED | REVIEW_REQUIRED | False |
| DynamicEarthNet | OFFICIAL_ARCHIVE_NOT_ACQUIRED | CC_BY_SA_4.0_OFFICIAL_RECORD_ARCHIVE_NOT_ACQUIRED | False |
| Forest-Change | PHYSICAL_READY_TEXT_HOLD | MIT_DATASET_CARD_WITH_ACADEMIC_REUSE_NOTE | False |
| GlobalGeoTree | NOT_REQUESTED | REVIEW_REQUIRED | False |
| Hi-UCD | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| LEVIR-MCI | TRAINING_ENABLED | SOURCE_TERMS_RECORDED_DATASET_REDISTRIBUTION_REVIEW_REQUIRED | False |
| NWPU-Captions | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| QAG-360K | NOT_REQUESTED | CC_BY_NC_4.0_MASK_DERIVED_RESEARCH_ONLY | False |
| RS5M | NOT_REQUESTED | REVIEW_REQUIRED | False |
| RSCC | ACCESS_REQUIRED | REVIEW_REQUIRED | False |
| RSCC-EBD | PHYSICAL_ONLY_HUMAN_REVIEW_REQUIRED | SOURCE_METADATA_AND_XBD_TERMS_RECORDED_REVIEW_REQUIRED | False |
| RSICD | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| RSITMD | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| RSRCC | PHYSICAL_ASSET_ACQUISITION_INCOMPLETE | APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED | False |
| S2Looking | PILOT_LOADER_VALIDATED | REVIEW_REQUIRED | False |
| SECOND-CC | TRAINING_ENABLED | SOURCE_TERMS_RECORDED_DATASET_REDISTRIBUTION_REVIEW_REQUIRED | False |
| SYSU-CD | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| SkyScript | NOT_REQUESTED | REVIEW_REQUIRED | False |
| SpaceNet-7 | OFFICIAL_ARCHIVE_NOT_ACQUIRED | REVIEW_REQUIRED | False |
| Synthetic-RCD-SECOND | DOWNLOAD_PARTIAL | REVIEW_REQUIRED | False |
| TAMMs | PHYSICAL_ONLY_TEXT_HOLD | APACHE_METADATA_FMOV_TERMS_AND_NONCOMMERCIAL_ANNOTATION_RESTRICTION | False |
| TERRA-CD | OFFICIAL_REPO_NO_RELEASE_ASSET | OFFICIAL_REPO_NO_RELEASE_ASSET_DATA_LICENSE_REVIEW_REQUIRED | False |

## Downloaded archives

- `Forest-Change`: `/mnt/weka/svardanyan/rs_change_project/datasets/raw/Forest-Change/Forest-Change-dataset.zip`, `23559661` bytes, SHA256 `424931a075f00f8cf21d4d2f622df688de559494844df4876b59bde13d3d855d`, status `ACQUIRED_PILOT_ARCHIVE`
- `TAMMs`: `/mnt/weka/svardanyan/rs_change_project/datasets/raw/TAMMs/waste_disposal.tar`, `282142720` bytes, SHA256 `2dc7038f2cb206e93b1cf4c1e177be8548994ad4ce9c61e0a64aa6f41a51fe94`, status `PILOT_ARCHIVE_ONLY_FULL_ARCHIVE_NOT_ACQUIRED`

## Verification

- Release contract: `True`
- Physical hash audit: `True`
- Cross-split leakage: `True`
- Cross-source overlap: `True`
- Mask-free integrity: `True`
- Checksums: `True`
- Acceptance status: `HOLD_DATA_QUALITY_GATES_REMAIN`

## Model Agent handoff

- Declared model SHA: `a047bff262839bdd7401905206d626f227a38048`; observed HEAD: `4aaa7649daa0079689b577cf63184ca01c05f4e7`; observed worktree clean: `False`
- Open requests: `['M2D-20260804-001', 'M2D-20260804-002']`
- Training remains unauthorized; semantic/localized/long-series limitations are explicit in shared contracts.

## Unresolved work

- RSCC-EBD: expected advisory AI audit is MISSING_INPUT and two-distinct-reviewer human promotion is incomplete
- Forest-Change: official split leaks components; scene-disjoint/caption provenance acceptance and human review remain incomplete
- TAMMs: 489 physical sequences only; full inventory/license/fMoW terms and generated text review remain incomplete
- Semantic primary gold: train/development semantic physical partitions and human verification are not available in this release
- Long-series: no candidate has explicit query_temporal_extent, relevant_frame_range, and temporal_direction; generated text is excluded
- RSRCC: physical acquisition is incomplete; available assets, missing assets, parent overlap, and full acquisition/human review remain incomplete (see source_reports/rsrcc_source_audit.json)
- Dubai-CC, DynamicEarthNet, SpaceNet-7, and TERRA-CD: official archive/license acquisition or release asset checks remain open

## Reusable lesson

Physical acquisition, metadata, and generated text are separate evidence classes. Promote only pinned assets with hashes and explicit parent-overlap audits; reuse a physical hash cache only when the physical registry SHA is identical; keep generated or mask-derived text evaluation-only until human promotion.
