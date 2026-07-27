# Dataset-v2 access ledger

These sources must not be replaced with third-party mirrors.  The build keeps
their records out of training until their official release, license and hash
are captured in `datasets/registry/download_registry.jsonl`.

- **ChangeChat-87k** — request the dataset link published by the official
  [ChangeChat repository](https://github.com/hanlinwu/ChangeChat); the code
  repository alone is not a dataset archive.
- **Synthetic RCD CNAM-CD** and **original CNAM-CD** — use the official
  Synthetic RCD release and its stated original-dataset source.  Confirm the
  CNAM-CD license before download.
- **Hi-UCD corrected release** — the official project provides data through
  request-gated cloud links; use the post-2025-11-01 corrected release.
- **CA-CDD, DisasterM3 and SkyScript** — second-wave sources; acquire only
  from the respective official project release and preserve the archive hash.
- **RS5M** — metadata or a bounded subset only; the full archive is explicitly
  out of scope for Dataset-v2 bootstrap.
