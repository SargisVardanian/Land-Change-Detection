# QCPR teacher taxonomy and checkpoint recovery decision

Date: 2026-07-16
Scope: terminology audit and read-only checkpoint recovery audit. No historical checkpoint was moved, deleted, evaluated, rendered, distilled from, or used for Stage B/C training.

## Status

The expected historical artifacts are absent:

- E0 global retrieval checkpoint: `/mnt/weka/svardanyan/rs_change_project/runs/qcpr_e0_20260711-015629/pilot/best_retrieval.pt`
- B checkpoint directory: `/mnt/weka/svardanyan/rs_change_project/runs/qcpr_v3_rc1_B100_75e6a7c_replacement_20260715-162718`

They were not found by a recursive, read-only file search of `/mnt/weka/svardanyan`, by an exact-name search, or in accessible archive, backup, and snapshot-named directories. Consequently, historical E0/B evaluation, rendering, distillation, and v3 Stage B/C runs are blocked.

## Three different meanings that must never be merged

### 1. Frozen pretrained text reference

This is the pretrained Jina v5 text backbone before the trainable text adapter.

```
caption
  -> frozen Jina v5
  -> base_text_embedding
     -> reference embedding for preservation or text-derived similarity
     -> trainable text adapter
  -> adapted_text_embedding
```

It is not an image-retrieval teacher checkpoint and does not supply image ground truth. Existing legacy variable names such as `teacher_text_embedding` refer to this detached, pre-adapter Jina representation. Future reporting must call it `base_text_embedding` or `frozen_pretrained_text_reference`.

### 2. Frozen historical visual/global teacher

This is a separately-instantiated, frozen retrieval model loaded strictly from the historical E0 checkpoint. It supplies pair embeddings, adapted text embeddings, and global retrieval-score preservation targets.

It is the only component that may be called the `historical_global_teacher`. It is unavailable while the E0 checkpoint is missing. It must never be silently replaced by pretrained-only weights, random initialization, or the new 10-step restart smoke checkpoint.

### 3. Text-derived semantic relevance targets

`semantic_teacher_relevance_matrix` builds a caption-to-caption similarity target from detached frozen-Jina text embeddings. It is useful for soft retrieval targets and false-negative exclusion, but is not image relevance ground truth, human structured relevance, or a model teacher checkpoint.

Future reports must call this family `text_derived_semantic_relevance` (or `semantic_pseudo_target`), not `teacher semantic relevance` without qualification.

## Architecture terminology

```
TEXT
caption
  -> frozen Jina v5
  -> base_text_embedding [frozen pretrained text reference]
  -> trainable text adapter
  -> adapted global text embedding + token embeddings

IMAGE
T1, T2
  -> frozen/partially frozen UniverSat
  -> temporal adapter
  -> global pair embedding + temporal patch field

ALIGNMENT
adapted global text <-> global pair
text tokens <-> temporal patches
```

If available, the historical global teacher is a separate frozen copy of the whole v1 text/image global-retrieval path. It is not the Jina backbone and not the text-derived pseudo-target calculation.

## Code and documentation audit

| Area | Current use of "teacher" | Correct classification |
| --- | --- | --- |
| `models/unichange_v2_retrieval.py` | `teacher_text_embedding` is detached pre-adapter Jina output | frozen pretrained text reference |
| `models/retrieval_heads.py`, `ucv2_retrieval_metrics.py`, smoke core | `semantic_teacher_relevance_matrix` and temperatures/top-k | text-derived semantic pseudo-target |
| `models/qcpr_v3_teacher.py`, `models/qcpr_v3_runtime.py`, `train_qcpr_v3.py` | `FrozenV1Teacher`, strict checkpoint load, preservation loss | historical global teacher; currently unavailable |
| `evaluate_qcpr_v3_fast.py`, `audit_qcpr_e0_full.py` | uses both a v1 checkpoint path and text-derived semantic targets | must label the two separately |
| legacy handoff and v2 docs | "semantic teacher" wording | terminology debt; must be clarified before future scientific reporting |

No code behavior was changed by this audit. Terminology renaming should be a separate reviewed change, because serialized metric keys and existing reports depend on current names.

## Recovery audit evidence

Read-only searches performed:

- recursive `find /mnt/weka/svardanyan` for `.pt`, `.pth`, `.ckpt`, and `.safetensors`;
- exact E0/B directory-name search;
- archive/backup/snapshot directory discovery;
- `~/.bash_history` review for QCPR paths and cleanup commands;
- QCPR project cleanup-code review.

The project has no generic cleanup/retention code that deletes historical QCPR run directories. The only relevant `rm -rf` is in smoke scripts and is limited to their explicitly supplied `RUN_DIR`. This does not establish who removed the historical directories; filesystem snapshot/backup recovery still requires storage administration.

Historical provenance that remains:

- exact historical source worktree: `/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-a78bf8bd`
- exact source commit: `a78bf8bd0d9079334a69749971842dcc8d142ece`
- E0 configuration: `/mnt/weka/svardanyan/rs_change_project/configs/qcpr_a78bf8bd/e0_qcpr_control.json`
- E0 pointer retained in: `configs/qcpr_a78bf8bd/latest_e0_run.txt`
- source manifests named by that configuration: `datasets/manifests/levir_mci.jsonl` and `datasets/manifests/second_cc_canonical.jsonl`.

The pointer is evidence of the intended E0 location, not evidence that the checkpoint remains accessible.

## Located model/checkpoint artifacts

| Classification | Path | Bytes | Mtime | SHA256 |
| --- | --- | ---: | --- | --- |
| pretrained text reference | `models/jina-v5-text-small-retrieval/model.safetensors` | 1,192,133,232 | 2026-06-29T14:18:53 | `0362107c2b13e18284c5152c4d4f667a4dec4665abdfe4f5e43a9bb799ba2276` |
| pretrained visual backbone | `models/universat-base/model.safetensors` | 804,264,576 | 2026-06-28T01:53:24 | `5cc062c9d4960ad405bca730765e7c23bc43c52471dbef0d821382990385be82` |
| archived unrelated RemoteCLIP | `archive/removed_models/remoteclip_20260628_015151/RemoteCLIP-ViT-B-32.pt` | 605,208,421 | 2026-06-22T01:10:32 | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` |
| E0 forensic score artifact, not model checkpoint | `runs/qcpr_v3_forensic_f3491ae6_20260715-175353/e0_full_natural/forensic_score_artifact.pt` | 112,255,613 | 2026-07-15T18:01:37 | `251b9084061ce75e5df895352e37d5ac51813723a6c2304ccbbbfe07dfa1abab` |
| new independent 10-step smoke, not historical E0 | `runs/qcpr_restart_v1_smoke_563872ea36b2_20260716-115016/best_retrieval.pt` | 2,349,654,304 | 2026-07-16T11:59:01 | `f47de80d9bd2012eb463638e14070679aa480f2abb071cd4b0efdad4ad298bb6` |
| new independent 10-step smoke, not historical E0 | `runs/qcpr_restart_v1_smoke_563872ea36b2_20260716-115016/last_retrieval.pt` | 2,349,654,304 | 2026-07-16T11:59:04 | `37da74b54ffe0b15560816c62a5f5ba4143e3a613c2e7bf59b4b2aa9379d2a25` |

The restart smoke also wrote checkpoint variants selected by individual short-run metrics. They are independent artifacts and must not be treated as historical E0 or used to make continuity claims.

## Recovery decision

### Option A: reproduce historical E0

Run a new, immutable reproduction from the exact preserved source commit, E0 config, and original manifests. Record every resolved setting, data fingerprint, seed, environment version, and output SHA256.

This can recreate the historical procedure, but cannot recreate the lost weights bit-for-bit unless all original runtime nondeterminism and job settings are recovered. Its output must be named `E0-reproduction`, never historical E0. It can become a future frozen historical-style global teacher only after independent validation and explicit acceptance.

### Option B: establish a clean independent baseline

Start from pretrained Jina v5 and UniverSat, with a new versioned manifest, fixed split fingerprints, and a documented global-retrieval training protocol. Name the resulting checkpoint `clean-baseline-v1`.

This option makes no continuity claim with the lost E0. It can be the global teacher for a new v3 branch only after its global retrieval baseline has passed the predefined gates. The restart smoke checkpoint is not sufficient: it only proves that the training path executes.

## Required decision before new v3 work

Choose A or B explicitly. Until then:

- no historical E0/B evaluation or rendering;
- no historical-global-teacher distillation;
- no v3 Stage B/C or query-mask experiment;
- no claim that text-derived semantic pseudo-targets are human-verified image relevance.

## Full inventory of independent restart-smoke checkpoints

These files were created by Slurm job 100866, not recovered from historical
storage. They are listed for provenance only and are not approved historical
or v3 teachers.

| Path suffix under `runs/qcpr_restart_v1_smoke_563872ea36b2_20260716-115016` | Bytes | Mtime | SHA256 |
| --- | ---: | --- | --- |
| `best_composite.pt` | 2,349,654,304 | 2026-07-16T11:58:59 | `6f29b835425feb8ad4443f579159d8a3c94724ec8ee2ee2031ec8937165b2395` |
| `best_detailed_r5.pt` | 2,349,656,958 | 2026-07-16T11:58:54 | `76dcd80bed425a18b684c1e202da6912ec56e97694a6caa4b368019c75d52ce1` |
| `best_macro_semantic.pt` | 2,349,660,843 | 2026-07-16T11:58:57 | `22a0b182f59e8b8005d611e4134a61f2691651d67667bc2890648ca84b5689f6` |
| `best_retrieval.pt` | 2,349,654,304 | 2026-07-16T11:59:01 | `f47de80d9bd2012eb463638e14070679aa480f2abb071cd4b0efdad4ad298bb6` |
| `best_semantic_ndcg10.pt` | 2,349,662,138 | 2026-07-16T11:58:52 | `21ad049a68e74bf3b298ef3f3889e4d17939a344a37729070247bb94bb428396` |
| `best_semantic_r1.pt` | 2,349,656,958 | 2026-07-16T11:58:48 | `7c24bd279c2f0d05f41260df72ab9528316bd50ffbeb105ec90612dee1835895` |
| `best_semantic_r5.pt` | 2,349,656,958 | 2026-07-16T11:58:50 | `aba165a8e3498f55ac02896e4742ea7257bea755e0589434ebf113428a669559` |
| `last_retrieval.pt` | 2,349,654,304 | 2026-07-16T11:59:04 | `37da74b54ffe0b15560816c62a5f5ba4143e3a613c2e7bf59b4b2aa9379d2a25` |

## Forensic correction: identified deletion event

After the original audit, the user supplied the recorded cluster-session transcript
for the cleanup on 2026-07-15. It explicitly reports:

- discovery of 42 run directories under
  `/mnt/weka/svardanyan/rs_change_project/runs`;
- preservation of only the active forensic directory
  `qcpr_v3_forensic_80225104_20260715-174118` and the completed forensic
  directory `qcpr_v3_forensic_f3491ae6_20260715-175353`;
- deletion of the other 40 run directories, said to free about 270 GB.

The expected E0 and B paths were both under `runs/` and were not in the
preserved list. Their current absence is therefore consistent with, and most
likely caused by, that cleanup event. This is stronger evidence than a
retention-policy hypothesis.

Correction to the earlier statement: project source code contains no generic
retention process that deleted these files, but an interactive cleanup action
did delete the historical run directories. Do not delete any additional
`runs/` directory. Recovery now requires a Weka snapshot or storage-admin
restore.
