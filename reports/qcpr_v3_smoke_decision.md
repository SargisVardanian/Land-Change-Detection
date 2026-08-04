# QCPR v3 bounded smoke decision

- Code SHA: `5eab23719aa78515383befd97145c49ffe5fc714`
- Job: `207556`, `COMPLETED`, exit `0:0`, node `gpu02`
- Allocation: one H100, `64G` host memory, 10-minute walltime
- Contract: 16 fixed optimizer steps; synthetic matrix `4 × 4`; no masks
- Exposure: 64 pair presentations and 64 query presentations; sequence hashes are in `exposure_accounting.json`
- Evidence path: finite gradients on all steps; minimum evidence gradient norm `3.6867e-05`
- Checkpoint SHA256: `a151c1f22237dabe0389109fd9bd922850f16eabacf411d4e26a4b632a0b53ee`
- Full-ranking SHA256: `6e7142ceb5b8485795681feebbb16eed7edf6a3014139a87783735ffddc99784`

The run is a mechanism/contract smoke, not a scientific retrieval result. CUDA
peak memory was not captured by this smoke script; the Slurm allocation was
`gpu:h100:1` with `64G` host memory. No mechanism pilot, P2, or main training
was submitted.

The earlier job `207555` is retained as an infrastructure-only failed attempt:
the driver reused an autograd graph between steps. It is not a model-quality
failure and was not promoted.
