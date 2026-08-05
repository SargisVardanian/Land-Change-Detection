# SigLIP-2 real integration smoke decision

The final bounded smoke is job `207706`, run
`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_siglip2_real_smoke_80d17d5_20260805`,
using code SHA `80d17d50845ac2a4086e32b68dd58c6371e92e27`.

It completed with exit `0:0` on an H100. The run used real LEVIR-MCI images,
the pinned `google/siglip2-base-patch16-256` checkpoint, real tokenization,
BF16 execution, eight optimizer steps, eight physical pairs, two captions per
pair, sixteen queries, and a single `16 x 8` relevance matrix. The native
outputs were 256 visual patches at hidden size 768 and 64 text tokens.

The smoke passed finite loss/gradients, frozen-backbone gradient isolation,
CUDA timing/memory instrumentation, and same-CUDA checkpoint round-trip. Peak
memory was 1.370 GiB allocated and 1.477 GiB reserved. The checkpoint SHA256 is
`d9f1fb6a387641bdd956725fa90da598fbea917bd4f72cbbbe3dde4b973513b5`.

This does not establish retrieval improvement. The evidence path is causal
under zeroing and query swapping, but the required deletion gate failed:
removing the top 10% evidence increased rather than decreased the score
(`score_drop_top = -0.01607`; bottom `= 0.00598`). Therefore
`SIGLIP2_REAL_INTEGRATION_SMOKE=PASS` but the mechanism remains
`BLOCKED_EVIDENCE_DELETION`. The batch did not contain explicit multi-positive
records, so that runtime path was not exercised.

No mechanism pilot, main training, P2 training, dataset download, or dataset
manifest change was performed.
