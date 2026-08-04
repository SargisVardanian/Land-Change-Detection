# QCPR Dataset-v2 final handoff — 2026-08-05

Decision: DATA_QUALITY_HOLD

## Provenance

- Branch: codex/qcpr-dataset-v2-final
- Release code SHA: b19cda128330ee8f67ab47dc39b3c81ad7740c7b
- Immutable release: /mnt/weka/svardanyan/rs_change_project/manifests/qcpr_dataset_v2_final_b19cda1_20260805
- Release SHA256SUMS digest: 74dc404f1d152cad516835d4febb6fcac2381f110f09365ff268dc7fdcddd81b (70 entries)
- Previous immutable releases were not modified.

## Counts and views

- Physical items: 32548; frames: 66074; sequences: 489
- Queries: 66452; training-enabled: 61080
- Query scopes: {'direction': 19015, 'exact': 43839, 'localized': 2000, 'semantic': 1598}
- Verification: {'derived_eval': 3598, 'human': 62854}

| View | train | development | test | status |
|---|---:|---:|---:|---|
| exact | 37529 | 5065 | 1245 | READY_EXACT_ONLY_CONDITIONAL_SOURCE_TERMS |
| semantic | 0 | 0 | 1598 | EVAL_ONLY_PRIMARY_GOLD_HOLD |
| localized | 0 | 0 | 2000 | EVAL_ONLY |
| direction | 16352 | 2184 | 479 | EXACT_CONDITIONAL |
| stable | 0 | 0 | 0 | EMPTY_GENERIC_NO_CHANGE_EXCLUDED |
| long_series | 0 | 0 | 0 | PHYSICAL_ONLY_NO_VALID_QUERY_METADATA_HOLD |

## Official source and license status

| Source | State | License | Training |
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
| RSRCC | METADATA_ACQUIRED_PHYSICAL_ASSET_AUDIT_PENDING | APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED | False |
| S2Looking | PILOT_LOADER_VALIDATED | REVIEW_REQUIRED | False |
| SECOND-CC | TRAINING_ENABLED | SOURCE_TERMS_RECORDED_DATASET_REDISTRIBUTION_REVIEW_REQUIRED | False |
| SYSU-CD | OFFICIAL_ARTIFACT_UNAVAILABLE | REVIEW_REQUIRED | False |
| SkyScript | NOT_REQUESTED | REVIEW_REQUIRED | False |
| SpaceNet-7 | OFFICIAL_ARCHIVE_NOT_ACQUIRED | REVIEW_REQUIRED | False |
| Synthetic-RCD-SECOND | DOWNLOAD_PARTIAL | REVIEW_REQUIRED | False |
| TAMMs | PHYSICAL_ONLY_TEXT_HOLD | APACHE_METADATA_FMOV_TERMS_AND_NONCOMMERCIAL_ANNOTATION_RESTRICTION | False |
| TERRA-CD | OFFICIAL_REPO_NO_RELEASE_ASSET | OFFICIAL_REPO_NO_RELEASE_ASSET_DATA_LICENSE_REVIEW_REQUIRED | False |

## Acquired archives

- Forest-Change: /mnt/weka/svardanyan/rs_change_project/datasets/raw/Forest-Change/Forest-Change-dataset.zip, 23,559,661 bytes, SHA256 424931a075f00f8cf21d4d2f622df688de559494844df4876b59bde13d3d855.
- TAMMs: /mnt/weka/svardanyan/rs_change_project/datasets/raw/TAMMs/waste_disposal.tar, 282,142,720 bytes, SHA256 2dc7038f2cb206e93b1cf4c1e177be8548994ad4ce9c61e0a64aa6f41a51fe94; pilot only.

## Verification

- Release-contract validator: PASS; 18 manifests, 32,548 items, 66,074 frames, 66,452 queries, 256 decode samples.
- Physical hash, cross-split, cross-source and mask-free audits: PASS; zero recorded violations.
- SHA256SUMS: PASS; 70 entries, no missing/extra/mismatch.
- Targeted dataset tests: 26 passed; full cluster suite was not run because the cluster runtime lacks torch.

## Model Agent handoff

- Declared model SHA: 4aaa7649daa0079689b577cf63184ca01c05f4e7; observed HEAD: 9d95181c3825c2cc49532d14adf1e628a6e746d3; observed worktree dirty: False.
- Model status: MODEL_V3_REAL_INTEGRATION_SMOKE_FAIL_POSTRUN_DIAGNOSTIC; dataset contract status: DATASET_CONTRACT_MISSING.
- Open requests: M2D-20260804-001, M2D-20260804-002.
- Main training, P1, P2-real, P2-semantic and P2-long-series remain unauthorized.

## Required reports and open work

- Named reports: source_reports/rsrcc_overlap_audit.json, source_reports/qag360k_overlap_audit.json, source_reports/localized_query_capabilities.json.
- Review packets: source_reports/human_review_packets/ with null decisions; status in source_reports/human_review_packet_status.json.
- RSCC-EBD AI audit remains MISSING_INPUT; two-reviewer agreement/adjudication is pending.
- Forest scene-disjoint/caption provenance acceptance is pending.
- TAMMs full inventory/license/event-disjoint split and verified text are pending.
- Semantic primary train/development, localized verified training rows, and validated long-series queries are absent.
- RSRCC remaining assets are blocked by HTTP 429; QAG-360K was not requested; Dubai-CC, DynamicEarthNet, SpaceNet-7 and TERRA-CD remain unacquired or unlicensed.

## Reusable lesson

Physical acquisition, metadata, and generated text are separate evidence classes. Promote only pinned assets with hashes and explicit parent-overlap audits; keep generated or mask-derived text evaluation-only until human promotion.
