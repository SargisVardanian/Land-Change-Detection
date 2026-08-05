# SigLIP-2 smoke decision — job 207780

The corrected bounded real integration smoke completed successfully:

- H100 `gpu02`, exit `0:0`;
- 16 optimizer steps;
- 8 physical pairs × 2 captions = 16 text queries;
- score matrix `16 × 8`;
- native visual output `[8,512,768]` and text output `[16,64,768]`;
- peak allocated `1.393 GiB`, peak reserved `1.650 GiB`;
- finite loss/gradients and frozen-backbone gradient isolation;
- checkpoint roundtrip PASS with maximum score difference `0.0`;
- checkpoint SHA256 `dc5ff006e6f416c0b65c8c181b2532fe907063af4f03f921338c9a51096f29c5`.

The BF16 masking overflow from job `207779` was fixed with a finite `-1e4`
sentinel. The complete run checksum manifest was then repaired to include the
final `smoke_summary.json`; the repair is recorded in the JSON decision.

This is a technical integration smoke, not a retrieval-quality result. The
development gallery was not evaluated and the multi-positive path was not
exercised. Evidence zeroing changed the score, but top-evidence deletion did
not pass (`-0.0230` versus bottom `0.0040`), so mechanism readiness remains
blocked. No evidence loss or other secondary objective was added.

```text
SIGLIP2_DOWNLOAD = VALID
SIGLIP2_SMOKE = PASS
SIGLIP2_MECHANISM = BLOCKED_EVIDENCE_DELETION
SIGLIP2_PHASE_A = NOT_AUTHORIZED
SIGLIP2_PHASE_B = NOT_AUTHORIZED
SIGLIP2_MAIN_MODEL = NOT_AUTHORIZED
P2_REAL/P2_SEMANTIC/P2_LONG_SERIES = BLOCKED_DATASET_CONTRACT
```
