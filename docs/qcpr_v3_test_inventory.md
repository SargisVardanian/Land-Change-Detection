# QCPR v3 test inventory

## Required functional coverage

| Contract | Planned test evidence |
|---|---|
| strict immutable v1 loading | real-checkpoint CPU load/shape audit plus mocked strict-load unit test |
| separate frozen teacher | distinct parameter identities, `eval`, no gradients, no optimizer membership |
| validated phase profiles | all five profiles; contradictory override rejection; resolved JSON |
| generic architecture | no semantic-specific output heads; exact tensor shapes; slots default off |
| canonical scoring parity | direct model, chunked API, trainer/evaluator/renderer adapters identical |
| faithful mask | identical canonical logits used for display, pooling, score and loss |
| multi-scale decoder | 32/16/8 inputs, 256×256 output, small target survives |
| mask losses | non-empty Dice/focal/Tversky; empty false-positive penalty; finite gradients |
| deterministic top-N | stable candidate selection, candidate recall and conditional rerank decomposition |
| dataset derivation | immutable inputs; deterministic outputs; no leakage; exact/near duplicate audit |
| sampler | duplicate-cluster aware; smoothed/capped weights; deterministic natural/balanced manifests |
| benchmark package | stratified deterministic selection; every label explicitly `unreviewed` |
| checkpoint | v3 state round-trip and config/hash preservation |
| numeric safety | no NaN/Inf and finite expected gradients |

During implementation only targeted tests are run. After the integrated patch,
the complete suite is run once with collection count and JUnit artifact at
`reports/qcpr_v3_pytest.xml`.

## RC1 complete-pass result

- collection: 404 tests;
- result: 403 passed, 1 skipped, 0 failed;
- elapsed: 301.34 seconds;
- immutable v1 checkpoint strict-load: executed and passed;
- compileall, shell syntax and `git diff --check`: passed;
- JUnit: `reports/qcpr_v3_pytest.xml`.
