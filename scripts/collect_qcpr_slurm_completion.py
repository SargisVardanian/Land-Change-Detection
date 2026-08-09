#!/usr/bin/env python3
"""Write one immutable completion snapshot for an upstream Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_record(job_id: str) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
            "-n",
            "-P",
        ],
        text=True,
    )
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    exact = [row for row in rows if row[0] == job_id]
    if len(exact) != 1 or len(exact[0]) < 5:
        raise RuntimeError(f"cannot resolve exact sacct row for {job_id}")
    row = exact[0]
    return {
        "job_id": row[0],
        "state": row[1],
        "exit_code": row[2],
        "elapsed": row[3],
        "node_list": row[4],
    }


def _artifact_inventory(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        if path.name == "completion.json" or path.name.endswith(".tmp"):
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
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    record = _job_record(args.job_id)
    payload = {
        "schema_version": "qcpr-slurm-completion-v1",
        "collected_at": datetime.now(UTC).isoformat(),
        "expected_code_sha": args.expected_code_sha,
        "upstream": record,
        "artifacts": _artifact_inventory(run_root),
    }
    target = run_root / "completion.json"
    temporary = run_root / "completion.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
