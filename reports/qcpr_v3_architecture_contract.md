# QCPR v3 model contract

The model branch is based exactly on 47a628f16aaefb98efa8a20d88f4dbe573dc7d47.

The implementation keeps the configured UniverSat and Jina backbones outside the
trainable temporal/text adapters. Native frame tokens are processed by one
joint variable-length temporal adapter with local cross-time displacement
attention, frame/sequence/change tokens, continuous timestamps and 2D
coordinates.

The final retrieval path has one scalar score:

global plus near-zero evidence plus near-zero change-slot plus near-zero late-interaction residual

The evidence weights are computed from query token to dense temporal token
similarities. The same weights produce the evidence vector used by the score.
The primary loss is one graded multi-positive listwise loss. No mask, direction,
segmentation or auxiliary loss is backpropagated.

The logical-batch implementation uses a memory-bounded token aggregate for a
Q-by-P score matrix. Single-candidate/Top-K forward retains the exact
query-token-by-token map for visualization and evaluation. The transition from
ANN stage 1 to evidence stage 2 is explicit; evidence is never computed over
the full gallery by rerank.

No semantic, localized or long-series dataset availability is inferred while
the Dataset Agent shared contract is missing.
