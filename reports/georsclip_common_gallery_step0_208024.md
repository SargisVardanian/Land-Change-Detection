# GeoRSCLIP frozen step-zero common-gallery evaluation

Status: `PASS_FROZEN_EVALUATION`.

This is an initialization reference only. No GeoRSCLIP parameter and no
temporal-head optimizer step was run.

## Contract

- Job `208024`, H100 `gpu01`, exit `0:0`.
- GeoRSCLIP revision:
  `4920188e6eba4e711ef9848cfd7cb77e874ee33f`.
- Checkpoint SHA256:
  `129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01`.
- Query count: `6,310`.
- Full physical gallery: `1,928`.
- Score matrix: `6310 × 1928`.
- Native visual tokens: `[1928,2,49,512]` (7×7 grid).
- Text tokens: `[6310,77,512]`.
- No masks or dense labels were accessed.

## Metrics

| Metric | Value |
|---|---:|
| Hit@1 | 0.000158 |
| Hit@5 | 0.001743 |
| Hit@10 | 0.003011 |
| Hit@50 | 0.023772 |
| Hit@100 | 0.051189 |
| Hit@500 | 0.280824 |
| Full-gallery MRR | 0.003683 |
| Mean rank | 914.344 |
| Median rank | 892 |

The positive and negative score means are `3.2537405` and `3.2537341`,
respectively. The untrained temporal/evidence head therefore provides no
scientific retrieval separation yet. Evidence maps are nearly uniform for
all queries; this is recorded as an initialization diagnostic.

## Resource use

- Peak allocated/reserved: `1.406 / 3.682 GiB`.
- Wall time: `63.35 s`.
- Optimizer steps: `0`.

The required 256-step GeoRSCLIP temporal-head baseline is not run in the
current bounded task. Its status remains `NOT_RUN`, not `VALID` or `INVALID`.

Full artifacts and SHA256SUMS:

`/mnt/weka/svardanyan/rs_change_project/runs/qcpr_georsclip_common_eval_step0_3615494_20260806`
