# Source provenance and readiness matrix

| Source | Local physical state | Text/provenance state | License/state blocker | Leakage/split state | Contribution | Decision |
|---|---|---|---|---|---|---|
| LEVIR-MCI / LEVIR-CC | 8,143 pairs, 16,286 frames | trusted human captions | official inventory mapping incomplete | exact/reverse/group split checks pass | diverse building change and exact retrieval | CORE READY |
| SECOND-CC | 4,701 pairs, 9,402 frames | trusted human captions | only partial official inventory represented | exact/reverse/group split checks pass | semantic land-cover change language and exact retrieval | CORE READY |
| Forest-Change | 334 pairs, 668 frames; physical inventory complete | five caption levels present; human/rule/generated lineage not sufficiently separated | caption provenance gate | scene-component-disjoint, zero component leakage reported | forest-specific/localized domain extension | HOLD |
| S2Looking | 1,000 pairs, 2,000 1024x1024 frames | no verified training text | localized text verification missing | dense masks eval-only | high-resolution physical geometry/localization evaluation | EVAL CANDIDATE, NOT READY |
| RSCC-EBD | 18,215 pairs, 36,430 frames | 0 verified rows; Reviewer A/B and adjudication incomplete | advisory audit/review missing | physical-only | large caption/grounding pool after factual rewrite | HOLD |
| TAMMs | 489 sequences, 1,956 frames | 0 verified long-series rows | license and temporal review unresolved | sequence-disjoint contract required | onset/duration/gradual/abrupt temporal supervision | HOLD |
| DUBAI-CC | no integrated assets | none | external acquisition/license/manifest absent | not auditable | urban change external-domain evaluation | NOT INTEGRATED |
| RSRCC | no integrated assets | parent provenance unresolved | license/parent lineage hold | not auditable | remote-sensing caption diversity | NOT INTEGRATED |
| DynamicEarthNet | no integrated assets | none | official archive/assets absent | not auditable | dense multi-temporal land-cover dynamics | NOT INTEGRATED |
| SpaceNet 7 | no integrated assets | none | official archive/assets absent | not auditable | long-series urban construction footprints | NOT INTEGRATED |
| TERRA-CD | no integrated assets | none | license and asset audit absent | not auditable | additional temporal/domain coverage | NOT INTEGRATED |

ChangeChat-87k, QAG-360K and other derived instruction corpora are not core sources.
They require parent-image lineage, license, physical hashes and factual-text verification
before any eligibility decision.

