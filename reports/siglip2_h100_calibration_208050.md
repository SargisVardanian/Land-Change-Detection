# SigLIP-2 H100 calibration — job 208050

The bounded calibration completed successfully on H100 `gpu05` with six
optimizer steps total: batch sizes 8, 16 and 32, two steps per candidate.
All candidates had finite gradients and no OOM.

| physical pairs | queries | peak allocated | peak reserved | status |
|---:|---:|---:|---:|---|
| 8 | 16 | 1.766 GiB | 1.957 GiB | PASS |
| 16 | 32 | 1.801 GiB | 2.342 GiB | PASS |
| 32 | 64 | 2.459 GiB | 2.711 GiB | PASS |

Selected contract:

```text
physical microbatch = 32 pairs
captions per pair = 2
queries per microbatch = 64
logical physical batch = 128
gradient accumulation = 4
```

The calibration used the Phase-B parameter scope only to measure memory. It is
not a Phase-B training result and makes no retrieval-quality claim. Phase A,
Phase B and GeoRSCLIP-256 remain unlaunched.

Run root:
`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_siglip2_h100_calibration_96c7366_20260806`
