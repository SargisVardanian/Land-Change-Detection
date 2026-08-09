# QCPR Dataset-v2

## Authoritative release

The authoritative immutable release is `qcpr_bitemporal_v2_train_20260808_final_r19g` at
`/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_bitemporal_v2_train_20260808_final_r19g`.
Its content-index SHA256 is
`6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`.
Code and documentation maintenance does not change this release or its scientific identity.

## Purpose and canonical model

Dataset-v2 supports temporal remote-sensing retrieval. It separates physical evidence
from language and task projections:

1. one physical item represents a pair or sequence;
2. native frames retain paths, hashes, dimensions, timestamps and nullable metadata;
3. one canonical query contains text, provenance, attributes, roles and exactly one
   retrieval-purpose classification;
4. sparse relevance edges encode graded query-to-item relevance;
5. collision groups identify ambiguous normalized captions and semantic neighbours;
6. immutable manifests project canonical records into task/split views;
7. masks and dense labels live only in evaluation sidecars.

Event IDs and source names never define semantic relevance. A query can have several
roles, but only one canonical classification.

## Inventory

| Source | Physical items | Frames | Current use |
|---|---:|---:|---|
| LEVIR-MCI / LEVIR-CC | 8,143 | 16,286 | exact core |
| SECOND-CC | 4,701 | 9,402 | exact core |
| RSCC-EBD | 18,215 | 36,430 | physical-only hold |
| S2Looking | 1,000 | 2,000 | high-resolution dense/eval candidate |
| Forest-Change | 334 | 668 | physical complete, text hold |
| TAMMs | 489 sequences | 1,956 | long-series hold |
| **Total** | **32,882** | **66,742** | mixed readiness |

There are 32,284 canonical queries. The immutable exact train/development/test views
contain 23,124 / 3,181 / 779 queries. Exact train covers 7,291 unique physical pairs:
3,214 LEVIR and 4,077 SECOND. All 23,124 exact-training rows are human-verified.

## Query views

| View | Train | Development | Test | Decision |
|---|---:|---:|---:|---|
| exact | 23,124 | 3,181 | 779 | READY |
| direction | 22,310 | 3,002 | 724 | EVAL_ONLY; training gate closed |
| localized | 0 | 228 | 2,230 | EVAL_ONLY |
| stable | 0 | 51 | 31 | EVAL_ONLY |
| long_series | 0 | 16 | 34 | HOLD |
| semantic | 0 | 0 | 0 | HOLD |

Direction rows overlap canonical exact captions as a projection. They are not an
additional independent text population. Semantic candidates contain 421 unverified
groups and must not be treated as training positives.

## Text policy

Canonical classifications are `exact_discriminative`, `semantic_multi_positive`,
`localized`, `direction_sensitive`, `stable_scene_specific`, `generic_no_change`,
`long_series`, and `unsupported_or_reject`. Classification uses provenance, normalized
collisions, attribute neighbours, physical evidence and review state; generic phrases
are only one weak signal.

The release disables 48,870 generic no-change rows and 11,301 unsupported/rejected
rows. Generic no-change captions are diagnostic-only. Exact training contains no
generic no-change, unsupported claims, mask-derived text, or generated-unverified text.

## Relevance and negatives

Exact grade 3 means the same verified physical item. All captions of one item are
mutual positives. Grades 1 and 2 and unresolved collision neighbours are ignored, not
negatives. Grade 0 is a verified negative; ordinary missing edges are implicit negatives.
The frozen 128x256 audit contains 256 positives, 613 ignores and 31,899 implicit
negatives, with zero same-pair false negatives.

Semantic grades are reserved as: 3 object + direction + specific change; 2 main
change category; 1 broad thematic similarity; 0 irrelevant. No semantic group is
training-enabled until human verification.

## Native geometry

Native dimensions are preserved: LEVIR and SECOND 256x256, Forest 480x480, RSCC and
TAMMs 512x512, S2Looking 1024x1024. All 66,742 frames decoded in the release audit.
GSD and footprint coverage are zero and remain null. Registration state is
`UNKNOWN_REGISTRATION` for all 32,882 physical items. Resize and NaFlex token-budget
selection are model-side operations; pair transforms must remain synchronized.

## Integrity and known limits

SHA256 and schema checks, exact/reverse pair leakage, physical-group split leakage,
native dimension consistency, mask isolation, and same-pair relevance passed in r19g.
The immutable 64-bit dHash audit covers all frames at Hamming threshold 4 within source.
An external final-audit supplement extends comparison across source and split boundaries:
4,208 different-item perceptual candidates, including 1,443 cross-split and 186
cross-source pairs, with zero exact-byte duplicates across either boundary. The 200
stored examples remain human-review candidates, not confirmed leakage. Human
false-negative rate and exact-scope precision are
`NOT_AVAILABLE` because 0/1,500 paper-calibration decisions are complete.

Core source accounting is incomplete relative to official inventories: LEVIR has
8,143 of 10,077 pairs and SECOND has 4,701 of 6,041. Missing/excluded reasons are not
fully machine-readable, so the release never silently restores those items.

## Training authorization

Only the LEVIR+SECOND exact core is authorized. Expanded/domain, semantic,
localized, stable, high-resolution and long-series training are not authorized.
See `reports/source_provenance_matrix.md` and `reports/dataset_limitations.md`.
