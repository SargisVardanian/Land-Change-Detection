# Dataset-v2 limitations

1. Exact-core paper calibration is incomplete: 0/1,500 required judgments. Exact-scope
   precision, 95% CI, false-exact rate and human semantic false-negative rate are not
   available.
2. LEVIR and SECOND inventories are partial. LEVIR has 8,143/10,077 and SECOND
   4,701/6,041 official pairs; 1,935 exclusions/missing mappings remain unresolved in
   the predecessor audit.
3. The 20.99% normalized duplicate rate and a collision group of size 1,361 make naive
   in-batch negative treatment unsafe.
4. Cross-source and explicit cross-split perceptual near-duplicate statistics are not
   present in the immutable release's within-source dHash artifact.
5. Source separability is unresolved. Batch quota balancing is not evidence that the
   model cannot exploit LEVIR/SECOND style.
6. GSD and footprint are unknown for every frame; registration is unknown for every
   physical item. Code must preserve null rather than infer metadata.
7. Forest caption-level provenance is not sufficient for training promotion.
8. RSCC has no completed dual review/adjudication and no verified factual caption set.
9. S2Looking masks are evaluation-only and cannot supply localized training text.
10. TAMMs license and long-series text verification remain unresolved.
11. Semantic groups are candidates only; no verified non-identical positives exist.
12. Stable/localized/direction views are evaluation projections, not authorization for
    their training losses.
13. DUBAI-CC, RSRCC, DynamicEarthNet, SpaceNet 7 and TERRA-CD are not integrated.
14. Random retrieval and complete per-source frozen baselines are not embedded in r19g.

These limits do not invalidate the LEVIR+SECOND exact core. They prohibit claims that
r19g is a retrieval-semantically complete expanded dataset.

