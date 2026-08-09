#!/usr/bin/env python3
"""Write one immutable completion snapshot for an upstream Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        is_completion = path.name in {
            "completion.json",
            "completion_accounting.json",
        }
        if is_completion or path.name.endswith(".tmp"):
            continue
        records.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--collector-job-id", default=None)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--required-artifact", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    collector_job_id = args.collector_job_id or os.environ.get(
        "SLURM_JOB_ID", "NOT_RUNNING_UNDER_SLURM"
    )
    artifacts = _artifact_inventory(run_root)
    present_paths = {str(item["path"]) for item in artifacts}
    required = {
        str(path): str(path) in present_paths for path in args.required_artifact
    }
    payload = {
        "schema_version": "qcpr-slurm-completion-v2",
        "timestamp": datetime.now(UTC).isoformat(),
        "upstream_job_id": args.job_id,
        "collector_job_id": collector_job_id,
        "run_root": str(run_root.resolve()),
        "expected_code_sha": args.expected_code_sha,
        "accounting_status": "NOT_AVAILABLE_ON_COMPUTE",
        "required_artifact_present": required,
        "required_artifacts_complete": all(required.values()),
        "artifacts": artifacts,
    }
    target = run_root / "completion.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        comparable_fields = (
            "schema_version",
            "upstream_job_id",
            "collector_job_id",
            "run_root",
            "expected_code_sha",
            "accounting_status",
            "required_artifact_present",
            "required_artifacts_complete",
            "artifacts",
        )
        if all(existing.get(key) == payload.get(key) for key in comparable_fields):
            return 0
        raise RuntimeError("completion.json exists with different immutable content")
    temporary = run_root / "completion.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
