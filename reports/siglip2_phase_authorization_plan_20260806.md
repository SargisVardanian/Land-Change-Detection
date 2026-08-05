# SigLIP-2 phase authorization plan — 2026-08-06

The real exact-core integration smoke is technically valid, but no quality
claim follows from it. The run used only LEVIR-MCI + SECOND-CC, with no masks,
RSCC text, generated-unverified captions, or Dataset-v2 edits.

The next scientific stages are prepared but intentionally not launched:

1. H100 batch calibration at physical microbatches 8, 16 and 32, two steps per
   candidate.
2. Frozen GeoRSCLIP temporal-head baseline for 256 steps.
3. SigLIP-2 Phase A temporal-head adaptation for 256 steps.
4. SigLIP-2 Phase B partial tower adaptation for 1,536 additional steps.

The current task explicitly limits GPU jobs to at most 32 steps, so these
training stages remain `NOT_AUTHORIZED`. The current smoke also did not run
the development gallery or exercise explicit multi-positive records. The top-
evidence deletion diagnostic failed; therefore no mechanism readiness or
retrieval improvement is claimed.

The Dataset Agent handoff requests remain open for verified semantic,
localized, directional, stable-scene and long-series views. These are not
required for the exact-core technical smoke but block P2 and later semantic
training.
