# SigLIP-2 real integration smoke: job 208013

Status: `PASS` (`0:0`), H100 `gpu05`, executable implementation SHA
`5aa11f16c4d163716bf5448f3725d5a4cab14070`.

The smoke used the authoritative Dataset-v2 HOLD release, exact LEVIR-MCI and
SECOND-CC rows only, real image decoding, the pinned local `google/siglip2-base-patch16-256`
checkpoint and the real tokenizer. It ran 8 BF16 optimizer steps with 8
physical pairs, 2 captions per pair, 16 queries, and a `16 x 8` score matrix.

The native visual contract was `[8, 2, 256, 768]`, the temporal dense output was
`[8, 512, 768]`, and text tokens were `[16, 64, 768]`. Both pretrained towers
were frozen; temporal adapter, evidence bottleneck and retrieval temperature
received finite gradients. Peak allocated/reserved VRAM was `2.001/2.031 GiB`.

Evidence diagnostics passed: query-swap map L1 `0.0003929`, query-swap score
change `0.4043`, evidence-zeroing score change `0.1665`, top-evidence score drop
`0.010586` versus bottom-evidence `0.008575`, detach gradient norm `0.3775`,
and time-reversal score change `0.005859`. Checkpoint roundtrip passed with
maximum score difference `0.0`.

The batch did not contain an explicit multi-positive query, so that runtime
case remains `MULTI_POSITIVE_RUNTIME_NOT_EXERCISED`. Full-gallery evaluation
was not run. This is an integration/mechanism smoke only and is not evidence
of retrieval-quality improvement.

Earlier attempts `208010` and `208011` remain preserved as engineering-only
driver failures (launcher mode and BF16/Float diagnostic comparison). They are
not scientific failures.
