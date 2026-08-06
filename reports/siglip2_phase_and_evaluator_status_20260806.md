# SigLIP-2 phase/evaluator status — 2026-08-06

Code is published at `e4e420628ed65a3ad6c03fb2c398746e40c0118a` on
`codex/qcpr-siglip2-temporal-training`; PR #7 remains draft and mergeable.

Implemented and CPU/static-tested:

- real mask-free image/text runtime helpers for the approved exact core;
- GradCache-style one-logical-matrix listwise training;
- explicit positive/ignored masks without caption-collision inference;
- per-module gradient and frozen-backbone audits;
- guarded Phase-A/B driver and Slurm launcher;
- common full-gallery evaluator plus Top-K evidence reranking;
- deterministic ranking, exposure and artifact hashes.

Validation: `639 passed, 3 skipped, 16 warnings`; focused SigLIP-2 contracts
`27 passed`; Pyright `0 errors, 0 warnings`; compileall, targeted Ruff, shell
syntax and diff-check all pass.

The latest bounded real smoke is job `207946` (`0:0`, 16 steps, H100): peak
allocated/reserved memory `1.393/1.650 GiB`, checkpoint roundtrip PASS,
evidence zeroing causal, temporal reversal measurable. Top-evidence deletion
still fails and explicit multi-positive runtime was not exercised. Therefore
the result is technical integration evidence only; no retrieval improvement is
claimed and mechanism training remains blocked.

Phase A/B and the common-gallery evaluation are implemented but not run.
The user’s current limit was respected: no job over 32 steps, mechanism pilot,
main training, GeoRSCLIP training or P2 was launched.
