# SigLIP-2 bounded real smoke — job 207946

Job `207946` completed with exit `0:0` on H100 `gpu05` using code SHA
`999eae4cb37067306b05c6b726f4ed871a72b2a1`.

## Contract

- 16 optimizer steps;
- 8 physical pairs and 2 captions per pair;
- 16 queries and a `16 × 8` score matrix;
- temporal visual tensor `[8, 512, 768]`;
- text tensor `[16, 64]`;
- only the approved LEVIR-MCI + SECOND-CC exact core;
- no masks, RSCC text, or generated-unverified text.

## Runtime

- peak allocated: `1.393 GiB`;
- peak reserved: `1.650 GiB`;
- CPU peak RSS: `3.318 GiB`;
- checkpoint roundtrip: PASS;
- no NaN/OOM: PASS;
- final checkpoint SHA256:
  `194bbb6b61e4fa7eb1ccb918b72bec97ef5a426df02afcd65c053b47789d97b6`.

## Mechanism diagnostics

- evidence zeroing changes the score by `1.25419` — evidence is causal;
- query swap changes the score by `0.00390625`;
- query-swap map L1: `0.00026685`;
- time reversal changes the maximum score by `0.0078125` — diagnostic passed;
- top-evidence deletion still fails: score drop `-0.01588`;
- bottom-evidence deletion score drop: `0.01301`;
- explicit multi-positive query: not exercised.

Therefore the technical smoke passes, but mechanism readiness remains blocked
by the top-evidence deletion criterion. Full-gallery retrieval evaluation was
not run and no model improvement is claimed.

The checksum manifest required a transparent post-run repair because Slurm
stderr received final process warnings after the Python process wrote its
in-process checksum. The repair changed only `SHA256SUMS`; the original
manifest hash is preserved in `artifact_repair.json`.
