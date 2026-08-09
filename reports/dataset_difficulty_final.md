# Final retrieval-data difficulty report

## Aggregate evidence

- 32,882 physical items, 66,742 frames, 32,284 canonical queries.
- Source physical shares: RSCC 55.39%, LEVIR 24.76%, SECOND 14.30%, S2Looking
  3.04%, TAMMs 1.49%, Forest 1.02%. This is not a uniform physical inventory.
- Exact-train query shares: SECOND 57.20%, LEVIR 42.80%.
- Exact-train pair shares: SECOND 55.92%, LEVIR 44.08%.
- Canonical normalized duplicate rate: 20.9925%.
- 1,278 collision groups; median size 2, mean 5.674, maximum 1,361.
- Exact-query identifiability: mean 0.6344, median 0.6542, range 0.3004-1.0.
- Generic no-change: 48,870 disabled rows, zero active exact-training contamination.
- Semantic candidates: 421 groups, zero verified groups.
- GSD/footprint known: 0/66,742 frames. Registration known: 0/32,882 items.

## Relevance difficulty

The frozen 128x256 matrix has 32,768 cells: 256 grade-3 positives (0.78125%),
613 collision/semantic ignores (1.8707%), 31,899 implicit negatives (97.3480%), and
zero same-pair false negatives. Mean positive-set size for exact core is one physical
item, while multi-caption rows of that item are mutual positives. Human semantic
false-negative rate is `NOT_AVAILABLE`; no immutable judgments support an estimate.

## Scope difficulty

| Scope | Main difficulty | Evidence state |
|---|---|---|
| exact | caption collisions, source shortcut, near-neighbour identifiability | train-ready core; paper precision calibration pending |
| direction | temporal reversal sensitivity | projection exists; independent direction gate closed |
| localized | spatial claim verification and evidence localization | eval-only rows, no verified training rows |
| stable | pair-discriminative anchors for no-change scenes | 82 eval rows, no verified train rows |
| semantic | non-identical positives and graded judgments | 421 candidates, 0 verified |
| long_series | onset/duration/gradual/abrupt semantics | 50 eval rows, 0 verified train rows |

## Baselines and unresolved measurements

The release links a frozen-anchor data-only comparison; a random retrieval baseline is
not present in r19g metadata. Existing visual dHash evidence decoded all frames with a
64-bit grayscale dHash at Hamming threshold 4, but comparisons are exhaustive only
within source. Cross-source and explicit cross-split perceptual candidate statistics
must be produced outside r19g before claiming a complete visual-overlap audit.

Balanced 64/64 LEVIR/SECOND sampling controls batch counts but does not prove source
invariance. A source-classification/separability test on frozen image and text features
is still required.

