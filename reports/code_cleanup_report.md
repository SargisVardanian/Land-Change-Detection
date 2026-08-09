# Dataset code cleanup report

## Changes

- Established `src/qcpr_data` as the single reusable Dataset-v2 implementation and
  documented ownership boundaries.
- Changed the final release validator to be read-only by default.
- The validator now rejects `--output` paths inside the release directory, preventing
  accidental mutation of immutable releases.
- Added a regression test for immutable-output rejection.
- Consolidated duplicate canonical SHA256 implementations; contracts and manifests now
  reuse `identities.hashing.sha256_file`.
- Restored the missing canonical decision-package helper required by the retained final
  Dataset-v2 builder, removing a real full-suite collection failure.
- Added a read-only cross-source/cross-split dHash auditor with an external-output guard;
  its evidence is stored under `runs/`, never in immutable r19g.
- Added canonical Dataset Card, Data Dictionary, source-provenance, difficulty and
  limitations reports.
- Classified historical builders as LEGACY_REQUIRED rather than deleting lineage.
- Identified the shared `handoff/dataset_final_to_model.json` as stale r18 transport
  state; it must be refreshed from r19g without modifying r19g itself.

## Preserved intentionally

Raw archives, immutable releases, human review artifacts, run artifacts, Git history,
historical release builders and model-side files were not removed or modified.
Release-specific paths in historical snapshots were not rewritten.

## Counts

Before cleanup: 681 non-cache files, about 89,983 lines. The cleanup adds documentation
and one focused safety test; it does not pretend to reduce historical lineage size.
Active canonical production code remains the `src/qcpr_data` package plus the final
validator. Generated untracked `__pycache__` directories in the cluster worktree are
safe to remove after validation and are not scientific changes.

## Remaining cleanup debt

- Historical builders duplicate orchestration and artifact-writing logic. Consolidating
  them physically would make old releases harder to reconstruct, so they remain
  LEGACY_REQUIRED.
- `audit_release()` in `manifests/integrity.py` writes builder-time audit artifacts and
  is retained only for legacy finalizers. It is not the final immutable audit path.
- A future major schema version may move legacy scripts under an archive namespace,
  but that is not justified in a no-scientific-change final pass.

Decision: cleanup is PARTIAL but safe. No new immutable release is required.
